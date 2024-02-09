from flask import Flask, render_template, request
from lib.Seq2Seq import Seq2SeqTransformer
from lib.Encoder import Encoder
from lib.Decoder import Decoder, DecoderLayer
from lib.Encoder_layer import EncoderLayer
from lib.Feed_foreward import PositionwiseFeedforwardLayer
from lib.Additive_attention import AdditiveAttention
from lib.Multihead_attention import MultiHeadAttentionLayer
from torchtext.data.utils import get_tokenizer
from nepalitokenizers import WordPiece

import torch

app = Flask(__name__)

SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'ne'

device = torch.device("mps")

token_transform = {}
text_transform = {}
vocab_transform = torch.load('./models/vocab.pt')

token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform["ne"] = WordPiece()

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            try:
                txt_input = transform(txt_input)
            except:
                # If an exception occurs, assume it's an encoding and use encode function
                txt_input = transform.encode(txt_input).tokens
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([2]), torch.tensor(token_ids), torch.tensor([2])))

for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform)



@app.route('/', methods=['GET', 'POST'])
def index():
    english_text = ''
    translation_result = ''
    if(request.method == 'POST'):
        english_text = request.form['english-text']

        print(english_text)
        
        params, state = torch.load('./models/additive_Seq2SeqTransformer.pt', map_location=device)
        model = Seq2SeqTransformer(**params, device=device).to(device)
        model.load_state_dict(state)
        model.eval()       

        # Tokenize and transform the input sentence to tensors
        input = text_transform[SRC_LANGUAGE](english_text).to(device)
        print(input)
        output = text_transform[TRG_LANGUAGE]("").to(device)
        input = input.reshape(1,-1)
        output = output.reshape(1,-1)

        # Perform model inference
        with torch.no_grad():
            output, _ = model(input, output)

        # Process the model output
        output = output.squeeze(0)
        output = output[1:]
        print(output)
        output_max = output.argmax(1)
        print("OutputMax",output_max)
        mapping = vocab_transform[TRG_LANGUAGE].get_itos()

        # # Save the input sentence to the list of previous queries
        # previous_queries.append(english_text)
        translation_result = []

        # Process the output tokens
        for token in output_max:
            token_str = mapping[token.item()]
            if token_str not in ['[CLS]', '[SEP]', '[EOS]','<eos>']:
                translation_result.append(token_str)
                print(translation_result)

        # Join the list of tokens into a single string
        translation_result = ' '.join(translation_result)

        print(translation_result)

    return render_template('index.html', translated_text = translation_result, english_text=english_text)

if __name__ == '__main__':
    app.run(debug=True)
