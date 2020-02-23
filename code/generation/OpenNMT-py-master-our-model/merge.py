import codecs

def delete_same(str_, pad):
    new_str = []
    str_words = str_.split(" ")
    i = 0
    if i + 2*pad  > len(str_words):
        return str_
    while i + 2*pad <= len(str_words):
        current_words = str_words[i:i+pad]
        i= i+pad
        while i + pad <= len(str_words) and " ".join(current_words) == " ".join(str_words[i:i+pad]):

            i = i+pad
        new_str+=current_words

    if i<len(str_words):
        new_str += str_words[i:]
    return " ".join(new_str)


def read_entity(path):
    sr = codecs.open(path, "r", "utf-8")
    lines = sr.readlines()
    entity_name = []
    for line in lines:
        line = line.strip()

        line = delete_same(line,1)
        line = delete_same(line,2)
        line = line.replace('"',"")
        entity_name.append(line)
    return entity_name



def read_template(path):
    sr = codecs.open(path, "r", "utf-8")
    sw = codecs.open("final_output.txt", 'w', 'utf-8')
    lines = sr.readlines()
    final_answer = []
    for i,line_item in enumerate(lines):
        line_item = line_item.strip()
        line = line_item.split(" <TSP> ")[0]
        entity1, entity2 = "", ""
        if len(line_item.split(" <TSP> ")) < 2:
            entity2 = ""
            entity1 = ""
        else:
            entity1 = line_item.split(" <TSP> ")[1]
         
           
            if len(line_item.split(" <TSP> ")) < 3:
                entity2 = ""
            else:
                entity2 = line_item.split(" <TSP> ")[2]

        if "<e>" in line:
            postion1 = line.index("<e>")
            if postion1+len("<e>") >= len(line):
                print(line)
            

            first = line[0:postion1+len("<e>")].replace("<e>",entity1)
            
            second = line[postion1+len("<e>"):].replace("<e>", entity2)
            final_answer.append(first + second)
            sw.write(first+second+"\n")
        else:
            sw.write(line+"\n")
    sw.close()

read_template("output.txt")
#read_template("multi-without-copy.txt", entity1, entity2)