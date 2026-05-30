
from transformers import AutoConfig

# class QwenClientInstance:
#     def __init__(self) -> None:
        




class QwenClient:
    def __init__(self):
        config = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B-FP8")
        print("config : ", config)

if __name__ == "__main__":
    c = QwenClient()


