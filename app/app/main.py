import litserve as ls

class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model1 = lambda x: x**2
        self.model2 = lambda x: x**3

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        squared = self.model1(x)
        cubed = self.model2(x)
        output = squared + cubed
        yield {"output": output}

    def encode_response(self, output):
        yield {"output": output}
    
    def unbatch(self,response):
        yield response

if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api,accelerator="auto")
    server.run(port=8000)



  