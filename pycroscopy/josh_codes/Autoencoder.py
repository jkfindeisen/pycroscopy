import os
import numpy as np

def load_data(data_path, resample_size=256):
    f = h5py.File(data_path, 'r')
    loop_data = f['filt_AI_mat'][()]
    X = np.rollaxis(loop_data.reshape(loop_data.shape[0], -1), 1)
    X_resample = np.zeros((X.shape[0], 40 * resample_size))
    for i in range(X.shape[0]):
        X_resample[i] = signal.resample(X[i], 40 * resample_size)
    X_resample = X_resample.reshape((-1, resample_size))
    X_resample -= np.mean(X_resample)  # TODO remove mean of each singal?
    X_resample /= np.std(X_resample)
    X_resample = np.atleast_3d(X_resample)

    return X_resample

def get_run_id(sample, layer_type, size, num_layers, embedding, lr, drop_frac, batch_size, kernel_size, **kwargs):
    run = (f"{sample}_{layer_type}{size:03d}_x{num_layers}_emb{embedding:03d}_{lr:1.0e}"
           f"_drop{int(100 * drop_frac)}_batch{batch_size}").replace('e-', 'm')
    if layer_type == 'conv':
        run += f'_k{kernel_size}'
    return run


def rnn_auto(layer, size, num_layers, embedding, n_step, drop_frac=0., bidirectional=True,
             **kwargs):
    if bidirectional:
        wrapper = Bidirectional
    else:
        wrapper = lambda x: x
    model = Sequential()
    model.add(wrapper(layer(size, return_sequences=(num_layers > 1)),
                        input_shape=(n_step, 1)))
    for i in range(1, num_layers):
        model.add(wrapper(layer(size, return_sequences=(i < num_layers - 1), dropout=drop_frac)))
    model.add(Dense(embedding, activation='linear', name='encoding'))
    model.add(RepeatVector(n_step))
    for i in range(num_layers):
        model.add(wrapper(layer(size, return_sequences=True, dropout=drop_frac)))
    model.add(TimeDistributed(Dense(1, activation='linear')))

    return model


def conv_auto(size, num_layers, embedding, n_step, kernel_size, drop_frac=0., **kwargs):
    """TODO: batch norm?"""
    model = Sequential()
    model.add(Conv1D(size, kernel_size, padding='same', activation='relu',
                     input_shape=(n_step, 1)))
    for i in range(1, num_layers):
        model.add(MaxPooling1D(2))
        model.add(Conv1D(size, kernel_size, padding='same', activation='relu'))

    model.add(Dense(embedding, activation='linear', name='encoding'))
#    model.add(Conv1D(embedding, kernel_size=1, activation='linear', name='encoding'))

    for i in range(1, num_layers):
        model.add(Conv1D(size, kernel_size, padding='same', activation='relu'))
        model.add(UpSampling1D(2))
    model.add(Conv1D(size, kernel_size, padding='same', activation='relu'))

    return model
