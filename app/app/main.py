from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from litserve import LitAPI, LitServer

class BERTLitAPI(LitAPI):
    def setup(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        self.model.to(device)
        self.model.eval()

    def decode_request(self, request):
        text = request["text"]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs

    def predict(self, inputs):
        with torch.no_grad():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        return outputs.logits

    def encode_response(self, logits):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        response = {
            "positive": probabilities[:, 1].item(),
            "negative": probabilities[:, 0].item()
        }
        return response

if __name__ == "__main__":
    api = BERTLitAPI()
    server = LitServer(api, accelerator='auto', workers_per_device=2)
    server.run(port=8000)
  