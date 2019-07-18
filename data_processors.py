import xml.etree.ElementTree
import os
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from functools import reduce
import numpy as np

tag_dict = {'POS': 0, 'NEG': 1}

class InputItem(object):
    """A single training/test example for simple sequence classification for applications like Winograd Challenge or PDP."""

    def __init__(self, guid, text_a, do_lower = True, text_b=None, label=None, groundtruth=None, decoy=None, reference_idx=None, tag=None):
        """Constructs a InputItem.

        Args:
            guid: string
                Unique id for the example.
            text_a: string 
                The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            do_lower: boolean
                Lower case the text, default = True
            text_b: (Optional) string. 
                The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. 
                The label of the example.
            groundtruth: (Optional) string
                True answer word
            decoy: (Optional) list
                List of distractor words
            reference_idx: (Optional) int
                Index of the pronoun in the text_b, for WNLI it is 0
            tag: (Optional) string
                Indicating the class of the item, e.g. POS for positive or NEG for negative
        """
        
            
        self.guid = guid
        self.text_a = text_a.lower().strip()
        self.text_b = text_b.lower().strip()
        self.label = label
        self.groundtruth = groundtruth
        self.decoy = decoy
        self.reference_idx = reference_idx
        self.tag = tag

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_items(self, data_dir):
        """Gets a collection of `InputItem`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputItem`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class XMLPDPProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_items(self, data_dir, select_type, do_lower=True):
        """See base class."""
        # WSCollection
        # PDPChallenge2016
        assert(select_type=='ambigious' or select_type=='resolved'), "Select extraction type either >ambigious< or >resolved<."
        return self._create_items( xml.etree.ElementTree.parse(os.path.join(data_dir, "PDPChallenge2016.xml")).getroot(), select_type, do_lower )


    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_items(self, xml_data, select_type, do_lower):
        sentences = []
        sent1 = []
        sent2 = []
        conjs = []
        prons = []
        choice0 = []
        choice1 = []
        examples = []
        guid_dict = dict()
        size = 0
        count_pos = 0
        count_neg = 0
        question = 0
        schema_count = 0
        for schema in xml_data.findall('schema'):
            schema_count+=1
            
            
            replas = ('\n', ''),('\n', '')
            replas2 = ('.', ''), ('.', ''), ('a ', ''), ('an ', ''), ('the ', ''), ('\'s','')
            
            
            sent1.append(reduce(lambda a, kv: a.replace(*kv), replas, schema[0][0].text.lower().strip()))
            sent2.append(reduce(lambda a, kv: a.replace(*kv), replas, schema[0][2].text.lower().strip()))
            prons.append(reduce(lambda a, kv: a.replace(*kv), replas, schema[0][1].text.lower().strip()))

            
            choices = []
            choices_ = []
            answers = len(schema[2])
            for i in range(0,len(schema[2])):
                choices.append(reduce(lambda a, kv: a.replace(*kv), replas,
                        schema[2][i].text.lower().strip()))
                choices_.append(reduce(lambda a, kv: a.replace(*kv), replas2,
                        schema[2][i].text.lower().strip()))
            
            

            ans = reduce(lambda a, kv: a.replace(*kv), replas2,
                        schema[3].text.lower().strip())
            
            ans_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g':6}
            
            decoys = []
            
            for i in range(0,answers):
                
                if not(ans_dict[ans] == i):
                    decoys.append(choices_[i])
                
            if select_type == 'resolved':
                for i in range(0,answers):


                    if ans_dict[ans] == i:
                        guid = (question)
                        label = '1'
                        text_a = sent1[size]
                        text_b = choices[i] + ' ' + sent2[size]
                        tag = tag_dict['POS']
                        examples.append(InputItem(guid=guid, text_a=text_a, text_b=text_b, label=label, groundtruth=choices_[i], tag=tag))
                        guid_dict[guid] = len(examples)-1
                        count_pos+=1
                    elif not(ans_dict[ans] == i):
                        guid = (question)
                        label = '0'
                        text_a = sent1[size]
                        text_b = choices[i] + ' ' + sent2[size]
                        tag = tag_dict['NEG']
                        examples.append(InputItem(guid=guid, text_a=text_a, text_b=text_b, label=label, groundtruth=choices_[ans_dict[ans]], tag=tag))
                        guid_dict[guid] = len(examples)-1
                        count_neg+=1
                        
            elif select_type == 'ambigious':
                guid = (question)
                label = '1'
                text_a = sent1[size]
                text_b = prons[size] + ' ' + sent2[size]
                tag = -1
                examples.append(InputItem(guid=guid, text_a=text_a, text_b=text_b, label=label, groundtruth=choices_[ans_dict[ans]],  decoy=decoys, tag=tag, reference_idx=0))
                guid_dict[guid] = len(examples)-1

            size += 1
            question += 1
        
        if select_type == 'resolved':
            print('Pos: '+str(count_pos) + ' Neg: '+str(count_neg)+ ' Sum: '+str(count_neg+count_pos))
        return examples, guid_dict
    
class XMLMnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_items(self, data_dir, select_type, do_lower=True):
        """See base class."""
        # WSCollection
        # PDPChallenge2016
        assert(select_type=='ambigious' or select_type=='resolved'), "Select extraction type either >ambigious< or >resolved<."
        return self._create_items( xml.etree.ElementTree.parse(os.path.join(data_dir, "WSCollection.xml")).getroot(), select_type, do_lower )


    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_items(self, xml_data, select_type, do_lower):
        sentences = []
        sent1 = []
        sent2 = []
        conjs = []
        prons = []
        choice0 = []
        choice1 = []
        examples = []
        guid_dict = dict()
        size = 0
        count_pos = 0
        count_neg = 0
        question = 0
        for schema in xml_data.findall('schema'):

            
        
            replas = (('\n', ''),('\n', ''))
            replas2 = (('.', ''), ('.', ''),(' a ', ''), ('\n', ''), (' an ', ''), ('the ', ''))
            
            sent1.append(schema[0][0].text.lower().strip())
            sent2.append(schema[0][2].text.lower().strip())
            prons.append(schema[0][1].text)

            c0 = reduce(lambda a, kv: a.replace(*kv), replas,
                        schema[2][0].text.lower().strip())
            c1 = reduce(lambda a, kv: a.replace(*kv), replas,
                        schema[2][1].text.lower().strip())
            
            
            c0_ = reduce(lambda a, kv: a.replace(*kv), replas2,
                        schema[2][0].text.lower().strip())
            c1_ = reduce(lambda a, kv: a.replace(*kv), replas2,
                        schema[2][1].text.lower().strip())


            ans = reduce(lambda a, kv: a.replace(*kv), replas2,
                        schema[3].text.lower().strip())
            
            if ans == 'a':
                
                if select_type == 'resolved':
                    guid = (question)
                    label = '1'
                    text_a = sent1[size]
                    text_b = c0 + ' ' + sent2[size]
                    tag = tag_dict['POS']
                    examples.append(InputItem(guid=guid, text_a=text_a, text_b=text_b, label=label, groundtruth=c0_, tag=tag))
                    guid_dict[guid] = len(examples)-1
                    count_pos+=1

                elif select_type == 'ambigious':
                    guid = (question)
                    label = '1'
                    text_a = sent1[size]
                    text_b = prons[size] + ' ' + sent2[size]
                    tag = -1
                    examples.append(InputItem(guid=guid, text_a=text_a, text_b=text_b, label=label, groundtruth=c0_,  decoy=[c1_], tag=tag, reference_idx=0))
                    guid_dict[guid] = len(examples)-1
                    
                
            else:
                if select_type == 'resolved':
                    guid = (question)
                    label = '1'
                    tag = tag_dict['POS']
                    text_a = sent1[size]
                    text_b = c1 + ' ' + sent2[size]
                    examples.append(InputItem(guid=guid, text_a=text_a, text_b=text_b, label=label, groundtruth=c1_, tag=tag))
                    guid_dict[guid] = len(examples)-1
                    count_pos+=1

                elif select_type == 'ambigious':
                    guid = (question)
                    label = '1'
                    text_a = sent1[size]
                    text_b = prons[size] + ' ' + sent2[size]
                    tag = -1
                    examples.append(InputItem(guid=guid, text_a=text_a, text_b=text_b, label=label, groundtruth=c1_,  decoy=[c0_], tag=tag, reference_idx=0))
                    guid_dict[guid] = len(examples)-1
                
                
            if select_type == 'resolved':    
                is_noun = lambda pos: pos[:2] == 'NN'
                tokenized = nltk.word_tokenize(text_a)
                nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
                nouns = list(filter(lambda a: a not in nltk.word_tokenize(c0_+' '+c1_), nouns))
                for k in range(0,len(nouns)):
                    guid = (question)
                    tag =tag_dict['NEG']
                    tmp = nouns[k] + ' ' + sent2[size]
                    examples.append(InputItem(guid=guid, text_a=text_a, text_b=tmp, label='0', tag=tag))
                    guid_dict[guid] = len(examples)-1
                    count_neg +=1

            size += 1
            question += 1
        
        if select_type == 'resolved':
            print('Pos: '+str(count_pos) + ' Neg: '+str(count_neg)+ ' Sum: '+str(count_neg+count_pos))

        return examples, guid_dict