import codecs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--data_path")
parser.add_argument("-v", "--task", choices=["task1","task3.1","task3.2"]) # task1, task3
args = parser.parse_args()
if args.task == "task3.1":
    train_file = "train_sml.txt"
    dev_file = "dev_sml.txt"
    test_file = "test_sml.txt"
elif arg.task == "task3.2":
    train_file = "train_sml_classifie.txt"
    dev_file = "dev_sml_classifie.txt"
    test_file = "test_sml_classifie.txt"
else:
    train_file = "train_classifier.txt"
    dev_file = "dev_classifier.txt"
    test_file = "test_classifier.txt"
    
def read(path):
    inputs = []
    targets = []
    sr = codecs.open(path, "r", "utf-8")
    lines = sr.readlines()
    sw_src = codecs.open(write_src, "w", "utf-8")
    sw_tgt = codecs.open(write_tgt, "w", "utf-8")
    for line in lines:
        line = json.loads(line)
        line["candidate predicate"] = line["candidate predicate"].split(" <split> ")
        for i in range(len(line["candidate predicate"])):
            line["candidate predicate"][i] = " ".join(line["candidate predicate"][i][4:].split("."))
            line["candidate predicate"][i] = line["candidate predicate"][i].replace("_"," ")
        items = " <SP> ".join([line["context"]] + line["candidate predicate"]) #, line["entity1"],line["entity2"]v items = " <SP> ".join([" ENT ".join([line["context"], line["entity1"],line["entity2"]])] + line["candidate predicate"]) 
        label = line["predicate"]
        input_ = items.strip().replace("<S>", "SEP")
        input_sentence = input_.split(" <SP> ")
        label = int(label)
        sw_src.write(input_sentence)
        sw_tgt.write(label)
    sw_src.close()
    sw_tgt.close()


base_path=args.data_path + "/"



data_path = args.data_path+"/"
read(data_path + train_file, data_path +"src-train.txt", data_path +"tgt-train.txt")
read(data_path + dev_file, data_path +"src-val.txt", data_path + "tgt-val.txt")
read(data_path + test_file, data_path +"src-test.txt", data_path +"tgt-test.txt")