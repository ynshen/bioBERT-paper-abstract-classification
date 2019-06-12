"""
Some utility function
"""


def progress_bar(pct):
    import sys
    sys.stdout.write('\r')
    # the exact output you're looking for:
    bar_num = int(round(pct * 20))
    sys.stdout.write("[%-20s] %d%%" % ('='*bar_num, pct*100))
    sys.stdout.flush()


def progress_checkpoint(proLen):
    return [round(proLen/50*i) for i in range(51)]


def load_pickle(dirc):
    import pickle
    try:
        with open(dirc, 'rb') as loadFile:
            data = pickle.load(loadFile)
    except FileNotFoundError:
        print('Error: file not found')
    try:
        with open(dirc + '.log') as filelog:
            print('%s:'%(dirc[dirc.rfind('/')+1:]))
            for line in filelog:
                print(line)
    except FileNotFoundError:
        pass
    return data


def dump_pickle(data, dirc, log, overwrite):
    import pickle
    import os
    
    flag = False
    if overwrite:
        flag = True
    elif os.path.isfile(dirc):
        overwrite=input('File exist, do you want to overwrite? (Y/N)')
        if overwrite in {'Y', 'y', 'yes', 'Yes'}:
            flag = True
    else:
        flag = True

    if flag:
        with open(dirc, 'wb') as dumpFile:
            pickle.dump(data, dumpFile)
        with open(dirc+'.log', 'w') as logFile:
            logFile.write(log)
        print('Data has been saved to %s' %dirc)


def extract_metadata(target, pattern):
    """
    Function to extract metadata info from target string (e.g. sample file name), provided pattern
    :param target: string to extract info
    :param pattern: pattern to extract metadata.
                    pattern rules:
                        [...] to include the region of sample_name,
                        {domain_name[, type]} to indicate region of domain to extract as metadata, including
                        [,int/float] will convert the domain value to np.int or np.float32 type, otherwise, string
                        e.g. R4B-1250A_S16_counts.txt
                             with pattern = "R4[{exp_rep}-{concentration, float}{seq_rep}_S{id, int}]_counts.txt"
                             returns metadata = {
                                     'name': 'B-1250A_S16'
                                     'exp_rep': 'B',
                                     'concentration': 1250.0,
                                     'seq_rep': 'A',
                                     'id': 16
                                    }
    :return: metadata
    """

    def extract_info_from_braces(target, pattern):
        """
        Iterative algorithm to extract metadata info from target and pattern
        """

        def stop(string, ix):
            """stop conditions when finding the rightest index of current domain(s)"""
            if ix == len(string) - 1:
                return True
            if string[ix] == '}' and string[ix + 1] != '{':
                return True
            else:
                return False

        def parse_domain(domain):
            """parse domain name and type"""
            if ',' in domain:
                domain = domain.split(',')
                if 'int' in domain[1] or 'i' in domain[1]:
                    return (domain[0], np.int)
                elif 'float' in domain[1] or 'f' in domain[1]:
                    return (domain[0], np.float32)
                else:
                    return (domain[0], str)
            else:
                return (domain, str)

        def get_domains(pattern):
            """
            inspect pattern and extract domain(s) from it
            multiple domains could be expected because of {}{} structure
            """

            domains = pattern[1:-1].split('}{')
            return [parse_domain(domain) for domain in domains]

        def extract_info(target, prefix, postfix):
            """
            extract substring in target are flanked by pre-fix and post-fix
            prefix: no need to find, always the first len(prefix) character of target
            postfix: the first occurrence of postfix substring after prefix, if not ''
            """
            if postfix == '':
                return target[len(prefix):]
            else:
                return target[len(prefix):target.find(postfix, len(prefix))]

        def divide_string(string):
            """
            split a string into consecutive chunks of digits, letters and upper letters
            """

            def letter_label(letter):
                if letter.isdigit():
                    return 0
                elif letter.isupper():
                    return 1
                else:
                    return 2

            label = [letter_label(letter) for letter in string]
            split_ix = [-1] + [ix for ix in range(len(string) - 1) if label[ix] != label[ix + 1]] + [len(string)]
            return [string[split_ix[i] + 1: split_ix[i + 1] + 1] for i in range(len(split_ix) - 1)]

        # anchor braces in pattern
        brace_left = pattern.find('{')
        if brace_left == -1:
            return {}
        brace_right = brace_left
        while not stop(pattern, brace_right):
            brace_right += 1
        domains = get_domains(pattern[brace_left:brace_right + 1])
        # find prefix and postfix
        prefix = pattern[:brace_left]
        postfix = pattern[brace_right + 1:pattern.find('{', brace_right + 1)
        if pattern.find('{', brace_right + 1) != -1 else len(pattern)]
        # anchor info domain in target from prefix and postfix
        info = extract_info(target, prefix, postfix)
        if len(domains) > 1:
            info = divide_string(info)
            if len(info) > len(domains):
                info[len(domains) - 1] = ''.join(info[len(domains) - 1:])
        else:
            info = [info]
        info_list = {domain[0]: domain[1](info[ix]) for ix, domain in enumerate(domains)}
        # interatively calculate the leftover substring
        if postfix != '':
            info_list.update(extract_info_from_braces(target=target[target.find(postfix, len(prefix)):],
                                                      pattern=pattern[brace_right + 1:]))
        return info_list

    metadata = {}  # dict to save extracted values
    # Anchor the position of brackets and curly braces in name_pattern
    brackets = [pattern.find('['), pattern.find(']')]
    metadata = extract_info_from_braces(target=target,
                                        pattern=pattern[:brackets[0]] + '{name}' + pattern[brackets[1] + 1:])
    metadata.update(extract_info_from_braces(target=metadata['name'],
                                             pattern=pattern[brackets[0] + 1: brackets[1]]))

    return metadata