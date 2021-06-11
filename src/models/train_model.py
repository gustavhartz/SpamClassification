import argparse
import transformers
import wandb
import sys
import torch
import tqdm

import torch.nn as nn
import torch.optim as optim


from transformers import AutoTokenizer
from src.models.model import LSTM


wandb.init()


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):

        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):

        # writer = SummaryWriter()
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument(
            "--lr", default=0.01, type=float, help="Learning rate for optimizer"
        )
        parser.add_argument(
            "--epoch",
            "-ep",
            default=2,
            type=int,
            help="Numper of epochs in training loop",
        )
        parser.add_argument(
            "--token_len",
            "--token_len",
            "-t",
            default=40,
            type=int,
            help="Max numbers of tokens in each string (pad or truncate to len)",
        )
        parser.add_argument(
            "--embedding_dim",
            "--embedding-dim",
            "-em",
            default=20,
            type=int,
            help="Dimension of embedding layer in LSTM",
        )
        parser.add_argument(
            "--hidden_dim",
            "--hidden-dim",
            "-h",
            default=128,
            type=int,
            help="Dimension of hidden layer in LSTM network",
        )
        parser.add_argument(
            "--dropout",
            "-d",
            default=0.5,
            type=float,
            help="Dropout rate for between LSTM layer and hidden layer",
        )
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # Text preprocessor
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        train_set = MNIST(
            "./data", train=True, download=False, transform=self.transform
        )
        train_set, val_set = torch.utils.data.random_split(train_set, [50000, 10000])

        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=64, shuffle=True
        )
        validloader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

        # Model
        model = LSTM(
            tokenizer.vocab_size,
            hidden_dimension=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            text_len=args.token_len,
        )

        # Magic
        wandb.watch(model, log_freq=100)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        num_epochs = args.epoch

        train_losses = np.zeros(num_epochs)
        valid_losses = np.zeros(num_epochs)

        writer_train_iter = 0
        writer_val_iter = 0

        loop = tqdm(range(num_epochs))
        for epoch in loop:
            trainiter = iter(trainloader)
            valiter = iter(validloader)

            model.train()
            train_loss = 0
            train_iters = 0

            for train_images, train_labels in trainiter:
                optimizer.zero_grad()
                output = model(train_images)
                batch_loss = criterion(output, train_labels)
                batch_loss.backward()

                optimizer.step()
                train_iters += 1
                train_loss += batch_loss

                # writer.add_scalar('Loss/train', batch_loss, writer_train_iter)
                # writer_train_iter+=1

                wandb.log({"train_loss": batch_loss})

            train_loss = train_loss / train_iters
            train_losses[epoch] = train_loss

            # EVALUTAION
            model.eval()
            val_loss = 0
            val_iters = 0

            # Accuracy measures
            val_preds, val_targs = [], []
            for valid_images, valid_labels in valiter:
                output = model(valid_images)
                val_batch_loss = criterion(output, valid_labels)
                val_loss += val_batch_loss

                # Softmax max arg
                preds = torch.max(output, 1)[1]

                val_targs += list(valid_labels.numpy())
                val_preds += list(preds.data.numpy())
                val_iters += 1

                # writer.add_scalar('Loss/val', val_batch_loss, writer_val_iter)
                # writer_val_iter+=1

                wandb.log({"val_loss": val_batch_loss})

            valid_acc_cur = accuracy_score(val_targs, val_preds)

            # writer.add_scalar('Accuracy/val', valid_acc_cur, epoch)

            val_loss /= val_iters
            valid_losses[epoch] = val_loss
            loop.set_postfix_str(
                s=f"Train loss = {train_loss}, Valid Loss = {val_loss}, Valid_Acc {valid_acc_cur}"
            )

        # writer.add_hparams({'lr':float(args.lr)},{'hparam/accuracy': valid_acc_cur})
        # writer.add_graph(model,valid_images)
        # writer.close()

        epoch = np.arange(num_epochs)
        plt.figure()
        plt.plot(epoch, train_losses, "r", epoch, valid_losses, "b")
        plt.legend(["Train Loss", "Validation Loss"])
        plt.xlabel("Updates"), plt.ylabel("Loss")
        plt.show()
        plt.savefig("./reports/figures")
        torch.save(
            model, "./models/Net/net_{}.model".format(len(os.listdir("models")))
        )  # creates subfolders

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--load_model_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        if args.load_model_from:
            model = torch.load(args.load_model_from)
        else:
            model = torch.load("./models/Net/net_1.model")

        test_set = MNIST(
            "./data", train=False, download=False, transform=self.transform
        )

        testloader = torch.utils.data.DataLoader(
            test_set, batch_size=256, shuffle=False
        )
        testiter = iter(testloader)

        test_preds, test_targs = [], []
        model.eval()
        for images, labels in testiter:
            output = model(images)

            # Softmax max arg
            preds = torch.max(output, 1)[1]
            test_targs += list(labels.numpy())
            test_preds += list(preds.data.numpy())

        print("Model test accuracy:", accuracy_score(test_targs, test_preds))


if __name__ == "__main__":
    TrainOREvaluate()
