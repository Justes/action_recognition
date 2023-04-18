import torch
import torch.nn as nn
# from torchsummary import summary

model_urls = {
    "c3d": "../pretrained/c3d-pretrained.pth"
}


class C3D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.__init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)


def init_pretrained_weights(model, model_url):
    """Initialiaze network."""
    corresp_name = {
        # Conv1
        "features.0.weight": "conv1.weight",
        "features.0.bias": "conv1.bias",
        # Conv2
        "features.3.weight": "conv2.weight",
        "features.3.bias": "conv2.bias",
        # Conv3a
        "features.6.weight": "conv3a.weight",
        "features.6.bias": "conv3a.bias",
        # Conv3b
        "features.8.weight": "conv3b.weight",
        "features.8.bias": "conv3b.bias",
        # Conv4a
        "features.11.weight": "conv4a.weight",
        "features.11.bias": "conv4a.bias",
        # Conv4b
        "features.13.weight": "conv4b.weight",
        "features.13.bias": "conv4b.bias",
        # Conv5a
        "features.16.weight": "conv5a.weight",
        "features.16.bias": "conv5a.bias",
        # Conv5b
        "features.18.weight": "conv5b.weight",
        "features.18.bias": "conv5b.bias",
        # fc6
        "classifier.0.weight": "fc6.weight",
        "classifier.0.bias": "fc6.bias",
        # fc7
        "classifier.3.weight": "fc7.weight",
        "classifier.3.bias": "fc7.bias",
    }

    #use_gpu = torch.cuda.is_available()
    #device = 'cuda' if use_gpu else 'cpu'

    #p_dict = torch.load(model_url, map_location=torch.device(device))
    p_dict = torch.load(model_url)
    model_dict = model.state_dict()
    for name in p_dict:
        if name not in corresp_name:
            continue
        model_dict[corresp_name[name]] = p_dict[name]
    model.load_state_dict(model_dict)
    print(f"Initialized model with pretrained weights from {model_url}")


def c3d_model(num_classes, pretrained=True, **kwargs):
    model = C3D(num_classes)
    if pretrained and kwargs.get("pretrained_model") != "":
        print(kwargs.get("pretrained_model", model_urls["c3d"]))
        init_pretrained_weights(model, kwargs.get("pretrained_model", model_urls["c3d"]))
    return model



if __name__ == "__main__":
    model = c3d_model(101)
    print(model)
    # summary(model, (3, 16, 112, 112))