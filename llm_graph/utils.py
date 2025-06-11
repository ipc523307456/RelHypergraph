import random
import tiktoken
import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

def evaluation_from_str(func_str, args_str):
    # Create a context dictionary for executing code.
    context = {}
    
    # Define the function.
    exec(func_str, context)
    
    # Extract the function name to invoke it dynamically.
    # This assumes that the function name is the word following "def".
    func_name = func_str.split()[1].split('(')[0]
    
    # Convert the args string into a dictionary.
    args_dict = eval(f"dict({args_str})", context)
    
    # Invoke the function with the parsed arguments.
    return context[func_name](**args_dict)

def num_tokens_from_messages(messages, model="gpt-4-0613"):
    """
    Return the number of tokens used by a list of messages.
    
    This function is copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb with slight revision.
    """
    
    
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
        
    # Revision
    if isinstance(messages, str):
        return len(encoding.encode(messages))
        
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def flatten_nested_dict(nested_dict, parent_key='', separator='_'):
    items = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_nested_dict(value, new_key, separator))
        else:
            items[new_key] = value
    return items

def merge_dicts(dict_list):
    merged_dict = {}

    for d in dict_list:
        for key, value in d.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    return merged_dict


def merge_nested_dicts(nested_dict_list):
    flat_list = [flatten_nested_dict(d) for d in nested_dict_list]
    merged_dict = merge_dicts(flat_list)
    return merged_dict

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag = tag if tag in ['A', 'N', 'V'] else 'N'  # Default to noun if not found
    return tag

def normalize_word(word):
    lemmatizer = WordNetLemmatizer()
    word = word.lower()
    pos = get_wordnet_pos(word)
    word = lemmatizer.lemmatize(word, 'v') if pos == 'V' else lemmatizer.lemmatize(word)
    return word

def remove_duplicates(input_list):
    res = []
    
    for x in input_list:
        if not x in res:
            res.append(x)
    return res

if __name__ == "__main__":
    print(normalize_word("seeing"))