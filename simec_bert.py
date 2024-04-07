import os
import torch
import string
from transformers import BertTokenizer, BertForMaskedLM
from jacobian_function import jacobian

#Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#-----------------------------------------------------------------------------------------


def load_model(model_name):
  """Load pre-trained model (either bert-base or bert-mini)

  Args:
      model_name: Either "bert-base" (768-d embedding space) or "bert-mini" (256-d embedding space).

  Returns:
      The loaded model.
  """
  try:
    if model_name.lower() == "bert-mini":
      bert_tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-mini')
      bert_model = BertForMaskedLM.from_pretrained('prajjwal1/bert-mini').eval()
      return bert_tokenizer,bert_model
    if model_name.lower() == "bert-base":
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
      return bert_tokenizer,bert_model
  except Exception as e:
    pass

#-----------------------------------------------------------------------------------------
  
#Decoder function
def decode(tokenizer, pred_idx, top_clean):
  """Decode a list of predicted ids to the corresponding tokens.

  Args:
      tokenizer: The tokenizer of the model
      pred_idx: A list of ids of the tokens to decode
      top_clean: How many predictions we want
  """
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])

#-----------------------------------------------------------------------------------------


def encode(tokenizer, text_sentence, add_special_tokens=True):
  """Returns the tuple input_ids, mask_idx containing the ids of the token given in input and of the mask token.

  Args:
      tokenizer: The tokenizer of the model.
      text_sentence (_type_): _description_
      add_special_tokens (bool, optional): _description_. Defaults to True.

  Returns:
      input_ids: A kist od ids corresponding to the input sentence.
      mask_idx : The id of the mask token.

  """
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx

#-----------------------------------------------------------------------------------------


def get_all_predictions(text_sentence, bert_tokenizer, bert_model, closest_vectors=5):
  """Given a sentence with a masked token, yields the top closest_vectors predictions

  Args:
      text_sentence: A sentence with a mask token to be predicted
      closest_vectors: The number of possibile predictions we want, in decreasing order of probability. Defaults to 5.

  Returns:
      #closest_vectors predictions.
  """
  input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
  #print(input_ids, mask_idx)
  with torch.no_grad():
    predict = bert_model(input_ids)[0]
  bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(closest_vectors).indices.tolist(), closest_vectors)
  return {'bert': bert}

#-----------------------------------------------------------------------------------------

def simec_bert(encoder, model_head, bert_tokenizer, delta, threshold, num_iter, embedded_input, eq_class_word_id, id_masked_word, print_every_n_iter):
  """Build a polygonal approximating the equivalence class of a token given an embedded input.

  Args:
      encoder: The encoder part of the model.
      model_head: The prediction head of the model.
      delta: The lenght of the segment we proceed along in each iteration.
      threshold: The threshold parameter we use to separate null and and non-null eigenvalues. 
        Below the threshold we consider an eigenvalue as null.
      num_iter: The number of iterations of the algorithm.
      embedded_input: The embedding of a sentence.
      eq_class_word_id: The id of the token of which we want to build the equivalence class.
      id_masked_word: The id of the masked word, which we want to keep constant.
      print_every_n_iter: The points built by the algorithm are printed every print_every_n_iter iterations.
  """

  
  #Build the identity matrix that we use as standard Riemannain metric of the output embedding space.
  embedding_dimension = embedded_input.shape[-1]
  g = torch.eye(embedding_dimension)

  #Clone and require gradient of the embedded input
  emb_inp_simec = embedded_input[:,:,:].clone()
  emb_inp_simec = emb_inp_simec.requires_grad_(True)

  #Compute the output of the encoder. This is the output which we want to keep constant
  encoder_output = encoder(emb_inp_simec)[0]

  #Send objects to GPU
  encoder = encoder.to(device)
  g = g.to(device)
  encoder_output = encoder_output.to(device)
  emb_inp_simec = emb_inp_simec.to(device)
  encoder_output = encoder(emb_inp_simec)[0]

  #Build an id-token dictionary, employed later to check the output tokens of the algorithm
  vocab_embedding = bert_tokenizer.vocab.values()
  id_to_token = {key : value for value,key in zip(bert_tokenizer.vocab.keys(),bert_tokenizer.vocab.values())}

  #Keep track of the length of the polygonal
  distance = 0.
  for i in range(num_iter): 

    #Compute the pullback metric
    jac = jacobian(encoder_output[0,id_masked_word,:], emb_inp_simec)[eq_class_word_id]
    jac_t = torch.transpose(jac, 0, 1)
    tmp = torch.mm(jac,g)
    pullback_metric = torch.mm(tmp,jac_t)

    #Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(pullback_metric, UPLO="U")

    #Select a random eigenvectors corresponding to a null eigenvalue.
    #We consider an eigenvalue null if it is below a threshold value-
    zero_eigenvalues = eigenvalues < threshold
    number_null_eigenvalues = torch.count_nonzero(zero_eigenvalues)
    id_eigen = torch.randint(0, number_null_eigenvalues, (1,)).item()
    null_vector = eigenvectors[:,id_eigen]

    #Proceeed along a null direction
    emb_inp_simec[0,eq_class_word_id,:] = emb_inp_simec[0,eq_class_word_id,:] + delta*null_vector
    distance += eigenvalues[id_eigen].item()*delta

    #Prepare for next iteration
    emb_inp_simec = emb_inp_simec.requires_grad_(True)
    encoder_output = encoder(emb_inp_simec)[0]

    if (i % print_every_n_iter == 0):
      tmp = encoder_output.cpu()
      similar_word = decode(bert_tokenizer, model_head(tmp)[0, eq_class_word_id, :].topk(5).indices.tolist(), 5)
      print(model_head(tmp)[0, eq_class_word_id, :])
      print("First five words in the equivalence class:")
      print(similar_word)
      print("Length of the polygonal in the embedding space :", distance)
      print("Argmax of the output:", torch.argmax(model_head(tmp)[0, id_masked_word, :]).item())
      print("Max of the output:", torch.max(model_head(tmp)[0, id_masked_word, :]).item())
      print("Output token:", id_to_token[torch.argmax(model_head(tmp)[0, id_masked_word, :]).item()])
      print("---------------------------------------------------------------")

#-----------------------------------------------------------------------------------------

def main():

  #Build the model
  model_name = "bert-mini"
  #model_name = "bert-base"
  bert_tokenizer, bert_model  = load_model(model_name) 

  #Get the embedding part of the model
  embedding = bert_model.bert.embeddings
  #Get the encoder part of the model
  encoder = bert_model.bert.encoder
  #Get the prediction head of the model, to convert the embedded output to a tokens
  model_head = bert_model.cls

  #Input sentence
  input_text = 'I would like a pie with'
  input_text += ' <mask>'
  #Check predictions
  prediction = get_all_predictions(input_text, bert_tokenizer, bert_model, closest_vectors=3)["bert"]
  print(prediction)
  print("---------------------------------------------------------------")

  #Build the embedding of the sentence
  input_ids, mask_idx = encode(bert_tokenizer, input_text)
  embedded_input = embedding(input_ids)

  #Set SiMEC parameters
  delta = 1.0
  threshold = 1e-2
  num_iter = 500
  print_every_n_iter = 10
  eq_class_word_id = 3
  eq_class_word_id = 5
  id_masked_word = 7

  #Run the algorithm
  simec_bert(encoder, model_head, bert_tokenizer, delta, threshold, num_iter, embedded_input, eq_class_word_id, id_masked_word, print_every_n_iter)

if __name__ == "__main__":
    main()