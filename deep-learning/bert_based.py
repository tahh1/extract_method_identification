from transformers import  AutoTokenizer, AutoModel, AutoModelForSequenceClassification,AutoModelForSeq2SeqLM
import torch, numpy as np

class Bert:

    def __init__(self, model_name) -> None:
        # model_name = "microsoft/graphcodebert-base"
        self.model_name=model_name
        if model_name == "microsoft/graphcodebert-base":
            self.tokenizer= AutoTokenizer.from_pretrained(model_name)
            self.model=AutoModel.from_pretrained(model_name)
        elif model_name == "microsoft/codebert-base":
            self.tokenizer= AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif model_name == "Salesforce/codet5-small":
            self.tokenizer= AutoTokenizer.from_pretrained(model_name)
            self.model=AutoModelForSeq2SeqLM.from_pretrained(model_name)            


    def generate_embeddings(self,code,device="cuda"):

        inputs = code.to(device)
        model = self.model.to(device)
        # print(inputs)
        # raise Exception


        outputs = model(inputs)
        # print(type(outputs))
        # print(outputs.__dict__)
        if self.model_name=="microsoft/graphcodebert-base":
            with torch.no_grad():
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        else:
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]  # Assuming the last layer's hidden state is required
            embeddings = last_hidden_state.mean(dim=1).squeeze()
        
        return embeddings



if __name__=="__main__":
    Bert().generate_embeddings()