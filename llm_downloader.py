from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
	filename="DeepSeek-R1-0528-Qwen3-8B-UD-Q8_K_XL.gguf",
)

llm.close()