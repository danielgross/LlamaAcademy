import os
import logging
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from omegaconf import OmegaConf
from ingest_docs import ingest_docs
from data_gen import launch_data_generation
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import argparse
from peft import prepare_model_for_int8_training
from utils import make_supervised_data_module, smart_tokenizer_and_embedding_resize

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = args_parse()
    cfg = OmegaConf.load(os.path.abspath(args.config))
    # Logging setup
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Ingest documents related to API
    api_docs = cfg.API_DOCS
    logger.info(
        "Indexing and embedding docs from {api}...".format(api=api_docs))
    
    if cfg.GENERATE:
        documents, documents_for_summary = ingest_docs(api_docs, recursive_depth=cfg.DEPTH_CRAWLING, logger=logger)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        logger.info(
            "Done indexing and embedding docs from {api}...".format(api=api_docs))

        logger.info("Code Generation...")
        if cfg.SUMMARIZE_DOCS:
            kwargs = {"documents_for_summary": documents_for_summary, "summary_embeds": True}
        launch_data_generation(
                            url_docs=api_docs,
                            documents_embeds=vectorstore,
                            output_dir=cfg.DATA_PATH,
                            num_tasks_to_generate=cfg.NUM_TASKS_TO_GENERATE,
                            model_name=cfg.OPENAI_ENGINE,
                            num_prompt_instructions=cfg.NUM_PROMPT_INSTRUCTIONS,
                            logger=logger,
                            **kwargs)
        logger.info("Done Generating Code...")

    gradient_accumulation_steps = cfg.BATCH_SIZE // cfg.MICRO_BATCH_SIZE
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        "jeffwan/vicuna-13b", load_in_8bit=True, device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        "jeffwan/vicuna-13b",
         model_max_length=2048,
         padding_side="right",
         use_fast=False
    )
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    })
    
    logger.info("Loaded model and tokenizer")
    train_dataset, eval_dataset, data_collator = make_supervised_data_module(tokenizer=tokenizer, data_path=cfg.DATA_PATH + "data.json")
    logger.info("Loaded dataset")

    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=cfg.MICRO_BATCH_SIZE,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=cfg.WARMUP_STEPS,
            num_train_epochs=cfg.EPOCHS,
            learning_rate=cfg.LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="no",
            output_dir=cfg.OUTPUT_DIR,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False
        ),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *
        _, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    logger.info("Training Process begins ...")
    trainer.train()
    model.save_pretrained(cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
