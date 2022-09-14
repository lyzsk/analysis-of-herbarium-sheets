from itertools import islice
import logging
from pathlib import Path

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from segmentation.instances import SemanticLabels

logging.basicConfig(format='[%(asctime)s] %(message)s', filename='training.log', filemode='w', level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def visualise_segmentation(predicted_class, colours):
    class_image = np.zeros((predicted_class.shape[1], predicted_class.shape[2], 3))
    prediction = predicted_class[0].cpu().numpy()
    for j in range(len(colours)):
        class_image[prediction == j] = colours[j]
    return class_image / 255


def visualise_results(output, original_image, reconstructed_image, predicted_class, colours, dpi=500):
    if not output.parent.exists():
        output.parent.mkdir()

    n = original_image.shape[0]
    gs = gridspec.GridSpec(3, n, width_ratios=[2.42] * n, wspace=0.05, hspace=0)
    plt.figure(figsize=(n * 2.42, 3))

    for i in range(n):
        plt.subplot(gs[0, i])
        plt.imshow(original_image.data[i].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.subplot(gs[1, i])
        plt.imshow(reconstructed_image.data[i].cpu().numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.subplot(gs[2, i])
        plt.imshow(visualise_segmentation(predicted_class[i], colours))
        plt.axis('off')
    plt.savefig(str(output), dpi=dpi, bbox_inches='tight')
    plt.close('all')


def torch_zip(*args):
    for items in zip(*args):
        yield tuple(item.unsqueeze(0) for item in items)


# **************************************************
# Train the segmentation network
# def train(model, instance_clustering, train_loader, test_loader, epochs, label_classes=5):
# **************************************************
def train(model, instance_clustering, train_loader, test_loader, epochs, label_classes=5):
    cross_entropy = nn.CrossEntropyLoss(weight=train_loader.labelled.dataset.weights.cuda())
    L2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    losses = {'train': {'semantic': [], 'instance': [], 'total': []},
              'test': {'semantic': [], 'instance': [], 'total': []}}
    accuracies = {'train': [], 'test': []}

    for epoch in range(epochs):
        scheduler.step()

        if epoch % scheduler.step_size == 0:
            logging.debug(f'Learning rate set to {scheduler.get_lr()[0]}')

        model.train()
        image = labels = instances = None

        for i, training_data in enumerate(train_loader):
            labelled = isinstance(training_data, tuple) or isinstance(training_data, list)

            if labelled:
                image, labels, instances = training_data
                image, labels, instances = Variable(image).cuda(), Variable(labels).cuda(), Variable(instances).cuda()
            else:
                image = training_data
                image = Variable(image).cuda()

            optimizer.zero_grad()

            z_hat1, x_hat, logits, instance_embeddings = model(image)
            z1 = model.forward_clean(image)[0]
            reconstruction_loss = L2(z_hat1, Variable(z1.data, requires_grad=False)) + L2(x_hat, image)
            loss = 20 * reconstruction_loss

            if labelled:
                logits_per_pixel = logits.view(image.shape[0], label_classes, -1).transpose(1, 2).contiguous()
                semantic_loss = cross_entropy(logits_per_pixel.view(-1, label_classes), labels.view(-1))

                instance_loss = sum(sum(instance_clustering(embeddings, target_clusters)
                                        for embeddings, target_clusters
                                        in SemanticLabels(image_instance_embeddings, image_labels, image_instances))
                                    for image_instance_embeddings, image_labels, image_instances
                                    in torch_zip(instance_embeddings, labels, instances))

                loss += semantic_loss * 10 + instance_loss

                predicted_class = logits.data.max(1, keepdim=True)[1]
                correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
                accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)

            loss.backward()
            optimizer.step()

            losses['train']['total'].append(loss.item())
            info = f'Epoch: {epoch + 1:{3}}, Batch: {i:{3}}, Loss: {loss.item()}'
            if labelled:
                info += f', Accuracy: {(accuracy * 100)}%'
            logging.info(info)
        epoch_name = f'epoch_{epoch + 1}'
        visualise_results(Path('results') / f'{epoch_name}.png', image, x_hat, predicted_class,
                          colours=train_loader.labelled.dataset.colours)
        np.save('losses.npy', [{'train': losses['train'], 'test': losses['test']}])
        np.save('accuracies.npy', [{'train': accuracies['train'], 'test': accuracies['test']}])

        epoch_name = f'epoch_{epoch + 1}'
        torch.save(model.state_dict(), Path('models') / f'epoch_{epoch + 1}')
        average_accuracy = testepoch(model, instance_clustering, test_loader, epoch_name)
        accuracies['test'].append(average_accuracy)
        accuracies['train'].append(accuracy)


# **************************************************
# Test segmentation on current epoch using test set
# use 50% of the labelled test set
# **************************************************
def testepoch(model, instance_clustering, test_loader, epoch_name):
    L2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    losses = {'train': {'semantic': [], 'instance': [], 'total': []},
              'test': {'semantic': [], 'instance': [], 'total': []}}
    accuracies = {'train': [], 'test': []}

    model.eval()

    image = labels = instances = None

    total_loss = 0
    total_accuracy = 0
    num_test_batches = 1
    with torch.no_grad():
        for i, testing_data in enumerate(test_loader.labelled.dataset):
            image, labels, instances = testing_data
            image, labels, instances = Variable(image.unsqueeze(0)).cuda(), Variable(labels).cuda(), Variable(
                instances).cuda()

            num_test_batches = i + 1

            z_hat1, x_hat, logits, instance_embeddings = model(image)
            z1 = model.forward_clean(image)[0]

            loss = L2(z_hat1, z1) + L2(x_hat, image)

            total_loss += loss.item()

            predicted_class = logits.data.max(1, keepdim=True)[1]
            correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
            accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)
            total_accuracy += accuracy

            # test 50% of the test set
            if i == (len(test_loader.labelled.dataset) / 2) - 1:
                break

    average_loss = total_loss / num_test_batches
    average_accuracy = total_accuracy / num_test_batches
    losses['test']['total'].append(average_loss)
    logging.info(
        f'Epoch: {epoch_name}, Batches: {num_test_batches}, Loss: {average_loss}, Accuracy: {(average_accuracy * 100)}%')

    return average_accuracy


# ***************************************************
# Evaluate segmentation on all epochs using test set
# use 100% of the labelled test set
# ***************************************************
def evaluateepochs(model, instance_clustering, test_loader, epochs, epochs_dir):
    # cross_entropy = nn.CrossEntropyLoss(weight=train_loader.labelled.dataset.weights)
    L2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    losses = {'train': {'semantic': [], 'instance': [], 'total': []},
              'test': {'semantic': [], 'instance': [], 'total': []}}
    accuracies = {'train': [], 'test': []}

    for epoch in range(epochs):
        # loads the epochs saved during training
        epoch_name = 'epoch_' + str(epoch + 1)
        epoch_file = epochs_dir + epoch_name
        model.load_state_dict(torch.load(epoch_file))
        model.eval()

        image = labels = instances = None

        total_loss = 0
        total_accuracy = 0

        num_test_batches = 1

        with torch.no_grad():
            for i, testing_data in enumerate(test_loader.labelled.dataset):
                image, labels, instances = testing_data
                image, labels, instances = Variable(image.unsqueeze(0)).cuda(), Variable(labels).cuda(), Variable(
                    instances).cuda()

                num_test_batches = i + 1

                z_hat1, x_hat, logits, instance_embeddings = model(image)
                z1 = model.forward_clean(image)[0]

                loss = L2(z_hat1, z1) + L2(x_hat, image)

                total_loss += loss.item()

                predicted_class = logits.data.max(1, keepdim=True)[1]
                correct_prediction = predicted_class.eq(labels.data.view_as(predicted_class))
                accuracy = correct_prediction.int().sum().item() / np.prod(predicted_class.shape)
                total_accuracy += accuracy
                # if i ==  len(test_loader.labelled.dataset)-1:
                #  break

        average_loss = total_loss / num_test_batches
        average_accuracy = total_accuracy / num_test_batches
        losses['test']['total'].append(average_loss)
        logging.info(
            f'Epoch: {epoch_name}, Batches: {num_test_batches}, Loss: {average_loss}, Accuracy: {(average_accuracy * 100)}%')
