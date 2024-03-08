
import os

max_seq_length = 32768

import torch
print(torch.cuda.is_available())

os.environ["WANDB_DISABLED"] = "true"

##################################################################################################################################
###choose only one of OPTION 1 or OPTION 2 to load the model and tokenizer########################################################
##################################################################################################################################



##################################################################################################################################
##################OPTION 1: create model for single or multiple GPU trainning using accelerate####################################
##################################################################################################################################

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


from accelerate import PartialState
device_string = PartialState().process_index


model_id = "mistralai/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # works with Llama models and reduces memory reqs
    device_map={'': device_string},

)



tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True,trust_remote_code=True, model_max_length=max_seq_length)


#setting up the tokenizer for unsloth and accelerate to be the same
tokenizer.pad_token = '<unk>'
tokenizer.add_bos_token = False
tokenizer.padding_side = "right"
print(tokenizer.pad_token)
print(tokenizer.add_bos_token)
print(tokenizer.padding_side)


###########lora model setup for accelerate################

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules = ["q_proj",
                      "k_proj",
                      "v_proj",
                      "o_proj",
                      "gate_proj",
                      "up_proj",
                      "down_proj",
                      ],
    lora_dropout=0,
    bias="none",
)

model = get_peft_model(model, peft_config)

#number of trainable parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))








##################################################################################################################################
##################OPTION 2: create model for UNSLOTH training#####################################################################
##################################################################################################################################
# from unsloth import FastLanguageModel
#
# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
#
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=model_id,
#     max_seq_length=max_seq_length,
#     dtype=None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#     load_in_4bit=False,
#
# )
#
# #setting up the tokenizer for unsloth and accelerate to be the same
# #set add BOS token to false
# tokenizer.add_bos_token = False
#
# print(tokenizer.pad_token)
# print(tokenizer.add_bos_token)
# print(tokenizer.padding_side)
#
#
#
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     lora_alpha=128,
#
#     target_modules = ["q_proj",
#                       "k_proj",
#                       "v_proj",
#                       "o_proj",
#                       "gate_proj",
#                       "up_proj",
#                       "down_proj",
#                       ],
#
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
# )
#
#
# #number of trainable parameters
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
#




#########################################################################
##################Data setup#############################################
#########################################################################

from datasets import load_dataset
dataset="cnn_dailymail"

data = load_dataset(dataset, '3.0.0')

#convert train data to pandas dataframe
import pandas as pd
train_df = pd.DataFrame(data['train'])

#keep first 10000 rows
train_df = train_df[:10000]

#create function to apply chat template to each row
def applyChatTemplate(row):
    #row = train_df.iloc[0]
    text=f"<s>[INST] Please summarize the following news article: \n\n{row['article']} [/INST] {row['highlights']} </s>"
    return text

#apply chat template to each row
train_df['text'] = train_df.apply(applyChatTemplate, axis=1)

textRow1 = train_df['text'][0]

#keep only the text column
train_df = train_df[['text']]

#create column of token lengths
train_df['token_length'] = train_df['text'].apply(lambda x: len(tokenizer.encode(x)))

#drop if token length is greater than max_seq_length
train_df = train_df[train_df['token_length'] <= max_seq_length]

del train_df['token_length']

#convert to dataset
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)

###########checking the tokenizer encoding and decoding##################
testText=train_dataset['text'][0]
encodeCheck = tokenizer(testText)
#decode to make sure it is correct
decodeCheck = tokenizer.decode(encodeCheck['input_ids']) #make sure there is only one bos token at the beginning





#########################################################################
##################training###############################################
#########################################################################


import transformers
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


training_arguments=transformers.TrainingArguments(
        save_steps=100,
        logging_steps=100,
        num_train_epochs=20,
        output_dir="./trainingResults",
        #evaluation_strategy="steps",
        do_eval=False,
        #eval_steps=5,
        #per_device_eval_batch_size=1,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=5,
        log_level="debug",
        # optim="paged_adamw_8bit",
        optim="adamw_torch",
        learning_rate=5e-5,
        # fp16=True,
        bf16=True, #use this for doing a full fine-tune without quantization, and with extra non-LoRA params enabled
        max_grad_norm=1,
        warmup_ratio=0.005,
        lr_scheduler_type="linear",
        gradient_checkpointing=True,  # Enable gradient checkpointing
        gradient_checkpointing_kwargs={'use_reentrant': False}  # Correctly passed arguments for gradient checkpointing

)



#setup to training on completions only
response_template = "[/INST]"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    #tokenizer=tokenizer,  #comment out if using collator
    model=model,
    train_dataset=train_dataset,
    #eval_dataset=test_dataset,
    args=training_arguments,
    data_collator=collator,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()











