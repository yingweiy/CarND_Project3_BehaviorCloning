from keras.models import load_model
from keras.utils import plot_model


model = load_model('m_nv1.h5')
plot_model(model, show_shapes=True, to_file='model.png')
print('Done.')

