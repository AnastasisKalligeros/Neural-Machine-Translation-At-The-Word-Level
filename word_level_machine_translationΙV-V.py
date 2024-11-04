import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
#                       CLASSES DEFINITION
# -----------------------------------------------------------------------------

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.positional_encoding = angle_rads[np.newaxis, ...]

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, _ = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.input_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.target_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding_input = PositionalEncoding(pe_input, d_model)
        self.pos_encoding_target = PositionalEncoding(pe_target, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.input_embedding(inp)
        enc_output += self.pos_encoding_input(enc_output)
        enc_output = self.dropout(enc_output, training=training)
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, training, enc_padding_mask)
        
        dec_output = self.target_embedding(tar)
        dec_output += self.pos_encoding_target(dec_output)
        dec_output = self.dropout(dec_output, training=training)
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)
        return final_output

# -----------------------------------------------------------------------------
#                       FUNCTIONS DEFINITION
# -----------------------------------------------------------------------------

def train_step(inputs, targets, transformer, optimizer, loss_fn, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        enc_padding_mask = transformer.create_padding_mask(inputs)
        look_ahead_mask = transformer.create_look_ahead_mask(tf.shape(targets)[1])
        dec_padding_mask = transformer.create_padding_mask(inputs)
        predictions = transformer(inputs, targets, True, enc_padding_mask, look_ahead_mask, dec_padding_mask)
        loss = loss_fn(targets, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(targets, predictions)

def evaluate_transformer(transformer, dataset, batch_size, max_length):
    total_bleu_score = 0
    num_sentences = 0
    smooth_fn = SmoothingFunction().method1

    for (input_seq, target_seq) in dataset.batch(batch_size):
        enc_padding_mask = transformer.create_padding_mask(input_seq)
        look_ahead_mask = transformer.create_look_ahead_mask(max_length)
        dec_padding_mask = transformer.create_padding_mask(input_seq)
        predictions = transformer(input_seq, target_seq, False, enc_padding_mask, look_ahead_mask, dec_padding_mask)
        
        for i in range(batch_size):
            reference = [target_seq[i].numpy().tolist()]
            candidate = tf.argmax(predictions[i], axis=-1).numpy().tolist()
            score = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)
            total_bleu_score += score
            num_sentences += 1

    return total_bleu_score / num_sentences

def train_model(num_epochs, delta_epochs, model, optimizer, data_preparation):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    bleu_scores = []
    losses = []

    for epoch in range(num_epochs):
        for (batch, (inputs, targets)) in enumerate(data_preparation.dataset):
            train_step(inputs, targets, model, optimizer, data_preparation.loss_fn, train_loss, train_accuracy)

        losses.append(train_loss.result())
        if epoch % delta_epochs == 0:
            bleu_score = evaluate_transformer(model, data_preparation.dataset, data_preparation.batch_size, data_preparation.max_length)
            bleu_scores.append(bleu_score)
            print(f'Epoch {epoch}: Loss = {train_loss.result():.4f}, BLEU Score = {bleu_score:.4f}')

        train_loss.reset_states()
        train_accuracy.reset_states()

    return bleu_scores, losses

# -----------------------------------------------------------------------------
#                       COMPARATIVE EVALUATION FUNCTION
# -----------------------------------------------------------------------------

def train_and_evaluate_models(models, data_preparation, num_epochs, delta_epochs):
    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        
        # Initialize optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        # Train the model
        scores, losses = train_model(num_epochs, delta_epochs, model, optimizer, data_preparation)
        
        # Store results
        results[model_name] = {
            "BLEU Scores": scores,
            "Losses": losses
        }

    return results

def plot_comparative_results(results):
    plt.figure(figsize=(14, 7))
    
    for model_name, metrics in results.items():
        plt.plot(metrics['Losses'], label=f'{model_name} Loss')
        plt.plot(metrics['BLEU Scores'], label=f'{model_name} BLEU Score')

    plt.legend(loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/BLEU Score')
    plt.title('Comparative Training Loss and BLEU Score over Epochs')
    plt.show()

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

# Assuming that DataPreparation is defined and prepares the data correctly
data_preparation = DataPreparation(DATAPATH, DATAFILE, SENTENCE_PAIRS, BATCH_SIZE, TESTING_FACTOR)

# Define models
models = {
    "RNN_Bahdanau": Decoder(data_preparation.french_vocabulary_size + 1, EMBEDDING_DIM, DECODER_DIM, attention_type='bahdanau'),
    "RNN_Luong": Decoder(data_preparation.french_vocabulary_size + 1, EMBEDDING_DIM, DECODER_DIM, attention_type='luong'),
    "Transformer": Transformer(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE, PE_INPUT, PE_TARGET)
}

# Train and evaluate models
results = train_and_evaluate_models(models, data_preparation, EPOCHS_NUMBER, DELTA_EPOCHS)

# Plot the results
plot_comparative_results(results)
