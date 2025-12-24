"""
Quantize the LoRA-finetuned model to 4bit.

Example:
    python3 finetune_quantize.py
"""

from pathlib import Path


CONFIG = {
    "lora_model_dir": Path("outputs/stage2"),
    "output_dir": Path("outputs/stage2_merged_4bit"),
    "max_seq_length": 4096,
    "load_in_4bit": True,
}


def main() -> None:
    try:
        from unsloth import FastModel
    except NotImplementedError as e:
        raise RuntimeError("Unsloth requires a GPU. Run this script on a CUDA machine.") from e

    lora_model_dir = CONFIG["lora_model_dir"]
    output_dir = CONFIG["output_dir"]

    if not lora_model_dir.exists():
        raise FileNotFoundError(f"LoRA model dir not found: {lora_model_dir}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(lora_model_dir),
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=CONFIG["load_in_4bit"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_4bit",
    )


if __name__ == "__main__":
    main()
