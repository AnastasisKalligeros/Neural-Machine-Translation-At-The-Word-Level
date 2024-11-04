# This is the main script file facilitating all the available functionality for
# the machine translation task at the word level.

# Import all required Python frameworks.
from classes.data_preparation import DataPreparation
from classes.encoder import Encoder
from classes.decoder import Decoder
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
#                       CLASSES DEFINITION
# -----------------------------------------------------------------------------
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)

    def call(self, query, values):
        score = tf.matmul(query, self.W(values), transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, values)
        return context_vector, attention_weights

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, attention_type='bahdanau'):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Επιλογή του τύπου προσοχής
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(self.dec_units)
        elif attention_type == 'luong':
            self.attention = LuongAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

# -----------------------------------------------------------------------------
#                       FUNCTIONS DEFINITION
# -----------------------------------------------------------------------------
def report_encoder_decoder():
    for encoder_in, decoder_in, decoder_out in data_preparation.train_dataset:
        encoder_state = encoder.init_state(data_preparation.batch_size)
        encoder_out,encoder_state = encoder(encoder_in,encoder_state)
        decoder_state = encoder_state
        decoder_pred,decoder_state,_ = decoder(decoder_in,decoder_state,encoder_out)
        break
    print("=======================================================")
    print("Encoder Input:           :{}".format(encoder_in.shape))
    print("Encoder Output:          :{}".format(encoder_out.shape))
    print("Encoder State:           :{}".format(encoder_state.shape))
    print("=======================================================")
    print("Decoder Input:           :{}".format(decoder_in.shape))
    print("Decoder Output           :{}".format(decoder_pred.shape))
    print("Decoder State            :{}".format(decoder_state.shape))
    print("=======================================================")

def loss_fn(ytrue, ypred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(ytrue, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = scce(ytrue, ypred, sample_weight=mask)
    return loss, loss

@tf.function 
def train_step(encoder_in, decoder_in, decoder_out, encoder_state):
    with tf.GradientTape() as tape:
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state, _ = decoder(decoder_in, decoder_state, encoder_out)
        loss = loss_fn(decoder_out, decoder_pred)
    variables = (encoder.trainable_variables + decoder.trainable_variables)
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

def predict(encoder, decoder):
    random_id = np.random.choice(len(data_preparation.input_english_sentences))
    print("Input Sentence: {}".format(" ".join(data_preparation.input_english_sentences[random_id])))
    print("Target Sentence: {}".format(" ".join(data_preparation.target_french_sentences[random_id])))
    encoder_in = tf.expand_dims(data_preparation.input_data_english[random_id], axis=0)
    encoder_state = encoder.init_state(1)
    encoder_out, encoder_state = encoder(encoder_in, encoder_state)
    decoder_state = encoder_state
    decoder_in = tf.expand_dims(tf.constant([data_preparation.french_word2idx["BOS"]]), axis=0)
    pred_sent_fr = []
    while True:
        decoder_pred, decoder_state, _ = decoder(decoder_in, decoder_state, encoder_out)
        decoder_pred = tf.argmax(decoder_pred, axis=-1)
        pred_word = data_preparation.french_idx2word[decoder_pred.numpy()[0][0]]
        pred_sent_fr.append(pred_word)
        if pred_word == "EOS" or len(pred_sent_fr) >= data_preparation.french_maxlen:
            break
        decoder_in = decoder_pred
    print("Predicted Sentence: {}".format(" ".join(pred_sent_fr)))

def evaluate_bleu_score(encoder, decoder):
    bleu_scores = []
    smooth_fn = SmoothingFunction()
    for encoder_in, decoder_in, decoder_out in data_preparation.test_dataset:
        encoder_state = encoder.init_state(data_preparation.batch_size)
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state, _ = decoder(decoder_in, decoder_state, encoder_out)
        decoder_out = decoder_out.numpy()
        decoder_pred = tf.argmax(decoder_pred, axis=-1).numpy()
        for i in range(decoder_out.shape[0]):
            target_sent = [data_preparation.french_idx2word[j] for j in decoder_out[i].tolist() if j > 0]
            pred_sent = [data_preparation.french_idx2word[j] for j in decoder_pred[i].tolist() if j > 0]
            target_sent = target_sent[0:-1]
            pred_sent = pred_sent[0:-1]
            bleu_score = sentence_bleu([target_sent], pred_sent, smoothing_function=smooth_fn.method1)
            bleu_scores.append(bleu_score)
    return np.mean(np.array(bleu_scores))

def clean_checkpoints():
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
    prefix = last_checkpoint.split("/")[-1]
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIRECTORY)]
    for file in checkpoint_files:
        status = file.find(prefix)
        if status == -1:
            if file != 'checkpoint':
                remove_file = os.path.join(CHECKPOINT_DIRECTORY, file)
                os.remove(remove_file)

def train_model(num_epochs, delta_epochs, encoder, decoder):
    eval_scores = []
    losses = []
    start_epoch = 1
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
    if last_checkpoint:
        clean_checkpoints()
        start_epoch = int(last_checkpoint.split("-")[-1])
        checkpoint.restore(last_checkpoint)
    finish_epoch = min(num_epochs, start_epoch + delta_epochs)
    for epoch in range(start_epoch, finish_epoch + 1):
        encoder_state = encoder.init_state(data_preparation.batch_size)
        for encoder_in, decoder_in, decoder_out in data_preparation.train_dataset:
            loss = train_step(encoder_in, decoder_in, decoder_out, encoder_state)
            print("Training Epoch: {0} Current Loss: {1}".format(epoch, loss[0].numpy()))
            eval_score = evaluate_bleu_score(encoder, decoder)
            print("Eval Score (BLEU): {}".format(eval_score))
            eval_scores.append(eval_score)
            losses.append(loss[0].numpy())
        predict(encoder, decoder)
        checkpoint.save(file_prefix=checkpoint_prefix)
        clean_checkpoints()
    return eval_scores, losses

# -----------------------------------------------------------------------------
#                       MAIN PROGRAM
# -----------------------------------------------------------------------------
DATAPATH = "datasets"
DATAFILE = "fra.txt"
SENTENCE_PAIRS = 15000
BATCH_SIZE = 64
TESTING_FACTOR = 10
CHECKPOINT_DIRECTORY = "checkpoints"
EPOCHS_NUMBER = 250
DELTA_EPOCHS = 30

data_preparation = DataPreparation(DATAPATH, DATAFILE, SENTENCE_PAIRS, BATCH_SIZE, TESTING_FACTOR)

EMBEDDING_DIM = 256
ENCODER_DIM, DECODER_DIM = 1024, 1024
encoder = Encoder(data_preparation.english_vocabulary_size + 1, EMBEDDING_DIM, ENCODER_DIM)
decoder = Decoder(data_preparation.french_vocabulary_size + 1, EMBEDDING_DIM, DECODER_DIM, attention_type='bahdanau')

checkpoint_prefix = os.path.join(CHECKPOINT_DIRECTORY, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

optimizer = tf.keras.optimizers.Adam()

report_encoder_decoder()

scores, losses = train_model(EPOCHS_NUMBER, DELTA_EPOCHS, encoder, decoder)

plt.plot(losses, label='Loss')
plt.plot(scores, label='BLEU Score')
plt.legend(loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Loss/BLEU Score')
plt.title('Training Loss and BLEU Score over Epochs')
plt.show()
