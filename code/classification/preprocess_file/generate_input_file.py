import codecs
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_path")
parser.add_argument("--task", choices=["task1","task3.1","task3.2"]) # task1, task3
args = parser.parse_args()
if args.task == "task3.1":
    train_file = "train_sml.txt"
    dev_file = "dev_sml.txt"
    test_file = "test_sml.txt"
elif args.task == "task3.2":
    train_file = "train_sml_classifier.txt"
    dev_file = "dev_sml_classifier.txt"
    test_file = "test_sml_classifier.txt"
else:
    train_file = "train_classifier.txt"
    dev_file = "dev_classifier.txt"
    test_file = "test_classifier.txt"

def read(path, write_src, write_tgt):
    sr = codecs.open(path, "r", "utf-8")
    lines = sr.readlines()
    sw_src = codecs.open(write_src, "w", "utf-8")
    sw_tgt = codecs.open(write_tgt, "w", "utf-8")
    print(path)
    for line in lines:
        line = json.loads(line)
        items = " <SP> ".join([line["context"],line["entity1"],line["entity2"]])
        label = line["label"].strip()

        sw_src.write(items+"\n")
        
        sw_tgt.write(label + "\n")
    sw_src.close()
    sw_tgt.close()


base_path=args.data_path

read(base_path + train_file, base_path +base_path +"src-train.txt", base_path +"tgt-train.txt")
read(base_path + dev_file, base_path +"src-val.txt", base_path +"tgt-val.txt")
read(base_path + test_file, base_path +"src-test.txt", base_path +"tgt-test.txt")



'''read_mask_input("train.txt", "src-train-maskinput.txt", "tgt-train-maskinput.txt")
read_mask_input("dev.txt", "src-val-maskinput.txt", "tgt-val-maskinput.txt")
read_mask_input("test.txt", "src-test-maskinput.txt", "tgt-test-maskinput.txt")




read_mask_A("train.txt", "src-train-maskA.txt", "tgt-train-maskA.txt")
read_mask_A("dev.txt", "src-val-maskA.txt", "tgt-val-maskA.txt")
read_mask_A("test.txt", "src-test-maskA.txt", "tgt-test-maskA.txt")



read_mask_B("train.txt", "src-train-maskB.txt", "tgt-train-maskB.txt")
read_mask_B("dev.txt", "src-val-maskB.txt", "tgt-val-maskB.txt")
read_mask_B("test.txt", "src-test-maskB.txt", "tgt-test-maskB.txt")



read_only_input("train.txt", "src-train-onlyinput.txt", "tgt-train-onlyinput.txt")
read_only_input("dev.txt", "src-val-onlyinput.txt", "tgt-val-onlyinput.txt")
read_only_input("test.txt", "src-test-onlyinput.txt", "tgt-test-onlyinput.txt")



read_only_A("train.txt", "src-train-onlyA.txt", "tgt-train-onlyA.txt")
read_only_A("dev.txt", "src-val-onlyA.txt", "tgt-val-onlyA.txt")
read_only_A("test.txt", "src-test-onlyA.txt", "tgt-test-onlyA.txt")



read_only_B("train.txt", "src-train-onlyB.txt", "tgt-train-onlyB.txt")
read_only_B("dev.txt", "src-val-onlyB.txt", "tgt-val-onlyB.txt")
read_only_B("test.txt", "src-test-onlyB.txt", "tgt-test-onlyB.txt")'''


