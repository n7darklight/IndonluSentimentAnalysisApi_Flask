from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import json

def getSentimentAnalysis(text=""):
    tokenizer = RobertaTokenizer.from_pretrained('model/tokenizer/')
    model = RobertaForSequenceClassification.from_pretrained('model/pre_trained/', num_labels=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    text = text.replace(r'[^\w\s]+','')
    text = text.lower()

    input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = input['input_ids'].to(device)
    attention_mask = input['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    output = torch.argmax(logits, dim=1).item()

    label = 'None'
    if output == 0:
        label = 'Negative'
    elif output == 1:
        label = 'Neutral'
    elif output == 2:
        label = 'Positive'

    result = {'Output': label}
    with open("files/result.json", "w") as outfile: 
        json.dump(result, outfile)
    print('Result done')
    
    if label != 'None':
        return {"results": "files/result.json", "type": "json", "status": 1, "message": "Success"}
    else:
        return {"results": "files/result.json", "type": "json", "status": 2, "message": "Failed"}