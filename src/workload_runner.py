import torch
import torchvision
import torchvision.transforms as transforms
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
PRECISION_MODE = args.precision
EPOCHS = 1
LEARNING_RATE = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./cifar_data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./cifar_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = torchvision.models.resnet50(pretrained=True)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler() if PRECISION_MODE == "fp16" else None

print("Starting training...")
start = time.time()
model.train()
epoch_loss, correct, total = 0.0, 0, 0

for batch_idx, (inputs, labels) in enumerate(trainloader):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    if PRECISION_MODE == "fp16":
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    epoch_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    if batch_idx % 20 == 0:
        print(f'Batch {batch_idx}, Loss: {loss.item()}')

end = time.time()
epoch_acc = 100. * correct / total
print(f'\nEpoch Loss: {epoch_loss / len(trainloader):.3f}')
print(f'Epoch Accuracy: {epoch_acc:.2f}%')
print(f'Training Time: {end-start:.2f} seconds')

model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        if PRECISION_MODE == "fp16":
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100. * test_correct / test_total
print(f'Test Accuracy: {test_acc:.2f}%')

with open('baseline_results.txt', 'w') as f:
    f.write(f'Epoch Accuracy: {epoch_acc:.2f}%\n')
    f.write(f'Test Accuracy: {test_acc:.2f}%\n')
    f.write(f'Training Time: {end-start:.2f} seconds\n')
