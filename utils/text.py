import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

import yaml
config = yaml.safe_load(open('config.yml'))
in_dir = config['in_dir']
out_dir = config['out_dir']

def join_strings(x, sep=', '):
    return sep.join(x)
    
def xml_to_dataframe(xml_file, format='jsonl', output_file = 'trial_data', in_dir = in_dir, ACD = False):
    """
    Parse an XML file and convert it to a pandas DataFrame and in the desired format.

    Args:
        xml_url (str): URL of the XML file or local file location
        format: 'jsonl' and 'tsv' accepted
        output_file: name of the file as output
        ACD: Set to True to also parse aspect category

    Returns:
        pandas.DataFrame: DataFrame containing the parsed data.
        file in the desired format in the 'in_dir' with the 'output_file' name
    """
    try:
        with open(xml_file, 'r') as f:
            xml_string = f.read()
    except:
        with urllib.request.urlopen(xml_file) as response:
            xml_string = response.read()

    root = ET.fromstring(xml_string)
    df = pd.DataFrame(columns=['id', 'prompt', 'completion'])

    for Review in root:
        for sentences in Review:
            prompt, completion = list(), list()
            for sentence in sentences:            
                for value in sentence:
                    if value.text != '\n                    ' and value.text != None:
                        prompt.append(value.text)           
                    for opinion in value:
                        if ACD:
                            completion.append(opinion.attrib['target']+'; '+opinion.attrib['category'].replace('#', ' ')+'; '+opinion.attrib['polarity'])
                        else:
                            completion.append(opinion.attrib['target']+'; '+opinion.attrib['polarity'])
            # Remove duplicates
            completion = list(dict.fromkeys(completion))
            # Extra formating for clarification for the model
            if len(completion) > 1:
                completion = '\n'.join(completion)
            else:
                completion = ''.join(completion)
            completion = ' '+completion+' END' # tokenization starts with a whitespace and ends with ' END'

            df = pd.concat([df,pd.DataFrame.from_dict({
                'id': Review.attrib['rid'],
                'prompt': (' ').join(prompt)+'\n\n###\n\n', # To separate prompt from completion
                'completion': completion,
                }, orient='index').T])
                            
    df.reset_index(inplace=True, drop=True)

    if format =='jsonl':
        df.drop(columns=['id']).to_json(os.path.join(in_dir, output_file+'.'+format), orient='records', lines=True)
    
    if format == 'tsv':
        df.drop(columns=['id']).to_csv(os.path.join(in_dir, output_file+'.'+format), sep="\t")
        #!openai tools fine_tunes.prepare_data -f os.path.join(in_dir, 'trial_data.tsv')
    
    df.to_parquet(os.path.join(in_dir, output_file+'.parquet'))
    return df

# Data preparation from openai format
def break_column_into_rows(df, column = 'completion', pattern = "\n"):
    new_rows = []   
    for index, row in df.iterrows():
        value = row[column]     
        if pattern in value:
            sub_rows = value.split(pattern)
            for sub_row in sub_rows:
                new_row = row.copy()
                new_row[column] = sub_row
                new_rows.append(new_row)
        else:
            new_rows.append(row)
    new_df = pd.DataFrame(new_rows)
    
    return new_df

def split_column(df, column_pattern =  ";|#", column = "completion"):
    """
    For the specific column (str) in the df (pd.DataFrame), it is split according to the column_pattern (regex)
    """
    new_cols = df[column].str.split(pat=column_pattern, expand=True)
    new_df = pd.concat([df.drop(column, axis=1), new_cols], axis=1)
    return new_df

# Data preparation for the graph_df
def capitalize_column_names(df):
    """
    Making the column names prettier for a graph
    """
    new_columns = df.columns.map(lambda x: x.replace('_', ' ').title())
    
    df = df.rename(columns=dict(zip(df.columns, new_columns)))
    return df

def capitalise_df(graph_df, cap_column = False):
  """
  Capitalise a dataframe
  """
  #Capitalising contents
  graph_df = graph_df.applymap(lambda x: x.replace('_', ' ').title().strip() if isinstance(x, str) else x)
  if cap_column:
    # Capitalising column names
    graph_df = capitalize_column_names(graph_df)
  return graph_df

  def join_strings(x, sep=', '):
    return sep.join(x)

def from_ai_to_df(file='data/trial_data.parquet', pattern = "\n", column_pattern =  ";|#", column = "completion", 
columns={0:'OTE', 1:"Entity", 2: "Attribute", 3:"Sentiment"}, cap_column = False, capitalise = True):
  """
  input: file location on parquet format or pandas DataFrame
  columns: dictionary of columns to be renamed
  """
  if isinstance(file, pd.DataFrame) == False:
    df = pd.read_parquet(file)
  else:
    df = file
  df[column] = df[column].apply(lambda x: x.replace(" END",""))
  if pattern: 
    df = break_column_into_rows(df, column, pattern)
  if column_pattern:
    df = split_column(df, column_pattern, column = column)
  if columns:
    df = df.rename(columns = columns)
  if capitalise:
    df = capitalise_df(df, cap_column = cap_column)
  return df

def amazon_date_to_iso(date):
    '''
    From the format '*m *d, yyyy' to 'yyyy-mm-dd'
    '''
    # Month
    m_= re.search(r'\s', date)
    m = date[0:m_.start()]
    if len(m)<2:
        m = '0' + m
    
    # Day
    d_ = re.search(r', ', date)
    d = date[m_.end():d_.start()]
    if len(d)<2:
        d = '0' + d

    # Year
    y = date[d_.end():]
    if len(y)<4:
        y = '20' + y
    return y+'-'+m+'-'+d

def xml_to_dataframe_task1(xml_file, format='jsonl', output_file = 'trial_data', in_dir = in_dir):
    """
    Parse an XML file from a URL and convert it to a pandas DataFrame.

    Args:
        xml_url (str): URL of the XML file.

    Returns:
        pandas.DataFrame: DataFrame containing the parsed data.
    """
    try:
        with open(xml_file, 'r') as f:
            xml_string = f.read()
    except:
        with urllib.request.urlopen(xml_file) as response:
            xml_string = response.read()

    root = ET.fromstring(xml_string)
    df = pd.DataFrame(columns=['id', 'prompt', 'completion'])

    for Review in root:
        for sentences in Review:
            for sentence in sentences:
                prompt, completion = list(), list()
                
                for value in sentence:
                    prompt.append(value.text)           
                    for opinion in value:
                        completion.append(opinion.attrib['target']+'; '+opinion.attrib['category']+'; '+opinion.attrib['polarity'])
                
                # Extra formating for clarification for the model
                completion.append(' END') # fixed stop sequence to inform the model when the completion ends
                if len(completion)>2:
                    completion = ' | '.join(completion)
                else:
                    completion = ''.join(completion)
                completion = ' '+completion # tokenization starts with a whitespace

                df = pd.concat([df,pd.DataFrame.from_dict({
                    'id': Review.attrib['rid'],
                    'prompt': prompt[0]+'\n\n###\n\n', # To separate prompt from completion
                    'completion': completion,
                    }, orient='index').T])
                            
    df.reset_index(inplace=True, drop=True)

    if format =='jsonl':
        df.drop(columns=['id']).to_json(os.path.join(in_dir, output_file+'.'+format), orient='records', lines=True)
    
    if format == 'tsv':
        df.drop(columns=['id']).to_csv(os.path.join(in_dir, output_file+'.'+format), sep="\t")
        #!openai tools fine_tunes.prepare_data -f os.path.join(in_dir, 'trial_data.tsv')
    
    df.to_parquet(os.path.join(in_dir, output_file+'.parquet'))

    return df

def xml_to_dataframe_task2(xml_url):
    """
    Parse an XML file from a URL and convert it to a pandas DataFrame.

    Args:
        xml_url (str): URL of the XML file.

    Returns:
        pandas.DataFrame: DataFrame containing the parsed data.
    """
    with urllib.request.urlopen(xml_url) as response:
        xml_string = response.read()

    root = ET.fromstring(xml_string)

    df = pd.DataFrame(columns=['id', 'text', 'category', 'subcategory', 'polarity'])

    for elem in root:
        row = []
        for child in elem:
            if child.tag == 'sentences':
                sentences=list()
                for grandchild in child:
                    for value in grandchild:
                        sentences.append(value.text)

            if child.tag == "Opinions":
                for value in child:
                    df = pd.concat([df,pd.DataFrame.from_dict({
                        'id': elem.attrib['rid'],
                        'text': ' '.join(sentences),
                        'category': value.attrib['category'].split('#')[0],
                        'subcategory': value.attrib['category'].split('#')[1],
                        'polarity':value.attrib['polarity']
                        }, orient='index').T])
    df.reset_index(inplace=True, drop=True)
    
    return df

def break_row_by_size(df, col_name='text', max_chars = 525):
    """
    Split any rows in the given column of the given Pandas DataFrame that are
    longer than the specified number of characters at the end of a sentence or
    a space.
    
    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        col_name (str): The name of the column to split.
        max_chars (int): The maximum number of characters allowed in each row.
    
    Returns:
        A new DataFrame with the specified column split into two if necessary.
    """
    new_df = pd.DataFrame(columns=df.columns)
    sentence_end_pattern = r"[.?!]"
    
    for index, row in df.iterrows():
        cell_contents = row[col_name]
        if len(cell_contents) > max_chars:
            # Split the cell contents at the end of a sentence or a space
            split_cells = re.findall(f"(.{{1,{max_chars}}}(?:(?:{sentence_end_pattern})|(?=\s)))\s?", cell_contents)
            new_row = row.copy()
            new_row[col_name] = split_cells[0]
            new_df = new_df.append(new_row)
            for split_cell in split_cells[1:]:
                new_row = row.copy()
                new_row[col_name] = split_cell.strip()
                new_df = new_df.append(new_row)
        else:
            new_df = new_df.append(row)
    
    return new_df

def report(y_test, y_pred, model_name, out_dir = out_dir, display_report = True, display_labels = None):
  ote_report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
  ote_report = pd.DataFrame(ote_report).transpose()
  ote_report.to_parquet(os.path.join(out_dir, model_name+'_report.parquet'))
  display(ote_report)

  if display_report:
    cm = confusion_matrix(y_test, y_pred, labels = display_labels)
    cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = display_labels)
    cm.plot()
    plt.savefig(os.path.join(out_dir, model_name+'_cm.png'))
    
    plt.show()
    return cm
  else:
    return ote_report
