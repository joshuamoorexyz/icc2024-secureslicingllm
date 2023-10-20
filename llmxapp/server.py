import os
from dataclasses import dataclass, field
from typing import Optional
from datasets.arrow_dataset import Dataset
from peft import PeftModel
from peft import LoraConfig, get_peft_model
import socket
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)

from trl import SFTTrainer

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-5)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.01)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=32)
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="iamtarun/python_code_instructions_18k_alpaca",
        metadata={"help": "The preference dataset to use."},
    )

    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=100,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=1000000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=50, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./results_packing",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def gen_batches_train():
    ds = load_dataset(script_args.dataset_name, streaming=True, split="train")
    total_samples = 10000
    val_pct = 0.1
    train_limit = int(total_samples * (1 - val_pct))
    counter = 0

    for sample in iter(ds):
        if counter >= train_limit:
            break

        original_prompt = sample['prompt']
        instruction_start = original_prompt.find("### Instruction:") + len("### Instruction:")
        instruction_end = original_prompt.find("### Input:")
        instruction = original_prompt[instruction_start:instruction_end].strip()
        content_start = original_prompt.find("### Output:") + len("### Output:")
        content = original_prompt[content_start:].strip()
        new_text_format = f'<s>[INST] {instruction} [/INST] {content}</s>'
        
        tokenized_output = tokenizer(new_text_format)
        yield {'text': new_text_format}

        counter += 1

def gen_batches_val():
    ds = load_dataset(script_args.dataset_name, streaming=True, split="train")
    total_samples = 10000
    val_pct = 0.1
    train_limit = int(total_samples * (1 - val_pct))
    counter = 0

    for sample in iter(ds):
        if counter < train_limit:
            counter += 1
            continue

        if counter >= total_samples:
            break

        original_prompt = sample['prompt']
        instruction_start = original_prompt.find("### Instruction:") + len("### Instruction:")
        instruction_end = original_prompt.find("### Input:")
        instruction = original_prompt[instruction_start:instruction_end].strip()
        content_start = original_prompt.find("### Output:") + len("### Output:")
        content = original_prompt[content_start:].strip()
        new_text_format = f'<s>[INST] {instruction} [/INST] {content}</s>'
        
        tokenized_output = tokenizer(new_text_format)
        yield {'text': new_text_format}

        counter += 1

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        quantization_config=bnb_config, 
        device_map=device_map, 
        use_auth_token=True,
        # revision="refs/pr/35" 
    )
    
    #### LLAMA STUFF 
    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1 
    # model.config.
    #### LLAMA STUFF 
    model.config.window = 256 

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        # target_modules=["query_key_value"], 
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM", 
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer







# Load the model, tokenizer, and other necessary components
model, peft_config, tokenizer = create_and_prepare_model(script_args)
ft_model = PeftModel.from_pretrained(model, "/home/dell-server2/Github/icc2024-secureslicingllm/llmxapp/mistral7B_finetune/train/alpaca-python-10k/results_packing/checkpoint-44200")
ft_model.eval()



# Define the server address and port
server_address = ('0.0.0.0', 12345)

# Create a socket and bind it to the server address
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(server_address)
server_socket.listen(1)  # Listen for incoming connections

#print("Server is listening on port 12345")


count=0

import time


while True:
    
    # Accept incoming connection
    print("Waiting for a connection...")
    client_socket, client_address = server_socket.accept()
    print("Accepted connection from:", client_address)

    # Receive JSON data from the client
    data = client_socket.recv(1024)
    if not data:
        break
    
    try:
        # Deserialize the JSON data
        received_data = json.loads(data.decode('utf-8'))
        
        # Process the received data (assuming it's a list of JSON objects)
        if isinstance(received_data, list):
            for metric in received_data:
                print("Received metric: Numeric Value =", metric["numericValue"], ", String Value =", metric["stringValue"])
                print(int(metric["numericValue"]))
                
                modelUEinput = int(metric["numericValue"])
                mockstring = "I recieved the data"
                startTime= time.time()
                #client_socket.send(mockstring.encode('utf-8'))
                #print("I sent the data")
                prompt = f"Based on 1 UE, generate KPI bounds in a C++ array"
                model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
                model.eval()
                # ft_model.eval()
                with torch.no_grad():
                    print("#"*50)
                    print("FT Model")
                    print("#"*50)
                    #print(tokenizer.decode(model.generate(**model_input, max_new_tokens=253, pad_token_id=2)[0], skip_special_tokens=True))
                    modelOutput = tokenizer.decode(model.generate(**model_input, max_new_tokens=253, pad_token_id=2)[0], skip_special_tokens=True)
                    client_socket.send(modelOutput.encode('utf-8'))
                    print("I sent the ", modelOutput)
                    elapsed_time = time.time() - startTime
                    print(f"Time elapsed: {elapsed_time*1000:.5f} milliseconds")
                    
                    print(f"I ran it {count+1} times")
                    count += 1

        else:   
            print("Received data is not a JSON array.")

    except json.JSONDecodeError as e:
        print("JSON decoding error:", str(e))

    client_socket.close()

server_socket.close()


# you load the model, tokenizer, and other necessary 
# components before entering the loop to receive and process 
# JSON data. You also convert the received metric data to a format 
# suitable for model input and generate text based on this input using 
# the fine-tuned model. The generated text is then printed. You would need 
# to define the specific logic for converting the metric data to the model_input 
# dictionary based on your requirements and the expected input format of the model.




# import socket
# import json
# import torch
# from transformers import PeftModel, PreTrainedTokenizerFast

# # Define the server address and port
# server_address = ('0.0.0.0', 12345)

# # Create a socket and bind it to the server address
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind(server_address)
# server_socket.listen(1)  # Listen for incoming connections

# print("Server is listening on port 12345")

# # Load the model, tokenizer, and other necessary components
# model, peft_config, tokenizer = create_and_prepare_model(script_args)
# ft_model = PeftModel.from_pretrained(model, "/root/results/checkpoint-150")
# ft_model.eval()

# while True:
#     # Accept incoming connection
#     print("Waiting for a connection...")
#     client_socket, client_address = server_socket.accept()
#     print("Accepted connection from:", client_address)

#     # Receive JSON data from the client
#     data = client_socket.recv(1024)
#     if not data:
#         break

#     try:
#         # Deserialize the JSON data
#         received_data = json.loads(data.decode('utf-8'))

#         # Assuming received_data is a list of JSON objects
#         if isinstance(received_data, list):
#             for metric in received_data:
#                 # Convert metric data to a format suitable for model input
#                 model_input = {}  # Define how to convert metric data to model input

#                 # Assuming you have some logic to convert metric data to model input

#                 # Generate text using the fine-tuned model and the converted input
#                 with torch.no_grad():
#                     generated_text = ft_model.generate(**model_input, max_new_tokens=512, pad_token_id=2)

#                 # Decode the generated text
#                 decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

#                 # Send the generated text back to the client
#                 client_socket.send(decoded_text.encode('utf-8'))

#         else:
#             print("Received data is not a JSON array.")

#     except json.JSONDecodeError as e:
#         print("JSON decoding error:", str(e))

#     client_socket.close()

# server_socket.close()

