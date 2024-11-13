import torch
from torchviz import make_dot
from torchview import draw_graph
from model import Model

if __name__ == '__main__':
    kernel_size = 3
    model = Model(kernel_size)

    model_graph = draw_graph(model, input_size=(8, 3, 600, 400), device='meta', roll=True)
    model_graph.visual_graph.render(filename=f'model_{kernel_size}', directory='./experiments', format='pdf')

    out = model(x=torch.arange(12 * 3 * 600 * 400).float().reshape(12, 3, 600, 400))
    make_dot(out, params=dict(list(model.named_parameters()))).render('rnn_torchviz', format='png')
