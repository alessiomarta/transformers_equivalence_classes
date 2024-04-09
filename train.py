import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from vit import ViT

LR = 5e-5
BATCH_SIZE = 128


def train_model(
    device,
    num_epochs,
    trainloader,
    image_size=28,
    channel_size=1,
    patch_size=7,
    embed_size=512,
    num_heads=8,
    classes=10,
    num_layers=3,
    hidden_size=256,
    dropout=0.2,
    measure_training_time=False,
    test_metrics=False,
    testloader=None,
):

    if measure_training_time:
        print("Training starts")
        start_time = time.time()
    model = ViT(
        image_size,
        channel_size,
        patch_size,
        embed_size,
        num_heads,
        classes,
        num_layers,
        hidden_size,
        dropout=dropout,
    ).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    loss_hist = {}
    loss_hist["train accuracy"] = []
    loss_hist["train loss"] = []

    for epoch in range(num_epochs):
        model.train()

        epoch_train_loss = 0

        y_true_train = []
        y_pred_train = []

        for _, (img, labels) in enumerate(trainloader):
            img = img.to(device)
            labels = labels.to(device)

            preds = model(img)

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
            y_true_train.extend(labels.detach().tolist())

            epoch_train_loss += loss.item()

        loss_hist["train loss"].append(epoch_train_loss)
        total_correct = len(
            [True for x, y in zip(y_pred_train, y_true_train) if x == y]
        )
        total = len(y_pred_train)
        accuracy = total_correct * 100 / total

        loss_hist["train accuracy"].append(accuracy)

        print("-------------------------------------------------")
        print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
        print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)
        print("-------------------------------------------------")

    if measure_training_time:
        print(f"--- Training ended in {(time.time() - start_time)} seconds ---")

    if test_metrics and testloader:
        y_true_test = []
        y_pred_test = []
        with torch.inference_mode:
            for _, (test_img, test_labels) in enumerate(testloader):
                test_img = test_img.to(device)
                test_labels = test_labels.to(device)
                test_pred = model(test_img)
                y_pred_test.extend(test_pred.detach().argmax(dim=-1).tolist())
                y_true_test.extend(test_labels.detach().tolist())
        total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x == y])
        total = len(y_pred_test)
        accuracy = total_correct * 100 / total

        print("---------------------------------------------------------")
        print(f"       Test Accuracy: {accuracy}=={total_correct}/{total}")
        print("---------------------------------------------------------")

    return model


if torch.backends.mps.is_available():
    dev = torch.device("mps")
else:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {dev.type}")


mean, std = (0.5,), (0.5,)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

trainset = datasets.MNIST("data/MNIST/", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = datasets.MNIST("data/MNIST/", download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

trained_model = train_model(
    device=dev,
    num_epochs=1,
    trainloader=trainloader,
    measure_training_time=True,
    test_metrics=True,
    testloader=testloader,
)

print("Saving model...")
torch.save(trained_model, "mnist_vit_trained.mdl")
