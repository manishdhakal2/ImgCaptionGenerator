
import torch
from models import CNN, LSTM

from loader import forward_cnn

from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


def eval_model(lstm_model, test_loader, encoder, device):
    lstm_model.eval()

    with torch.no_grad():
        all_references = []
        all_predictions = []

        progress = tqdm(test_loader)

        for batch_index, batch in enumerate(progress):
            img_features = batch[0].to(device)
            input_label = batch[1].to(device)
            final_word = batch[2].to(device)

            final_caption = input_label[:,1:].cpu().tolist()

            caption_words = encoder.decode_text(final_caption)

            for cap in caption_words:

                all_references.append(caption_words)  

            for img in range(len(img_features)):
                starting_token = torch.tensor([[encoder.word2idx["<START>"]]]).to(device)
                words = []

                while len(words) < 50:
                    logits = lstm_model(starting_token, img_features[img].unsqueeze(0))
                    last_word_index = torch.argmax(logits[:, -1, :], dim=1)
                    last_word = encoder.idx2word[last_word_index.item()]

                    if last_word == "<END>":
                        break

                    words.append(last_word)
                    starting_token = torch.cat([starting_token, last_word_index.unsqueeze(1)], dim=1)

                all_predictions.append(words)


            progress.set_postfix({'batch': batch_index})
        


        all_references = clean_reference(all_references)
        #print(f"Reference : {all_references[-1]}")
            

        #print(f"Predictions : {all_predictions[-1]}")
        print(type(all_references[0]))
        print(all_references[0])


        smoother = SmoothingFunction().method1
        scores = []
        for ref, hyp in zip(all_references, all_predictions):
            score =sentence_bleu([ref], hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother)
            scores.append(score)

        bleu4 = sum(scores) / len(scores)
        print(f"BLEU-4: {bleu4:.4f}") 



def clean_reference(caption_list):
    references = []
    for batch in caption_list:        # loop over batches
        for caption in batch:         # loop over images in batch
            clean = []
            for token in caption:
                if token in ["<PAD>", "<START>"]:
                    continue
                if token == "<END>":  # stop at first END, ignores duplicate
                    break
                clean.append(token)
            references.append(clean)
    return references








            

    



