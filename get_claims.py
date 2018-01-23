from collections import Counter
import re

fname = 'bbc.txt'

patterns = [
    r'([A-Z][a-z]+ [A-Z][a-z]+) (said) (.+?)\.',
    r'([A-Z][a-z]+) (said) (.+?)\.',
    r'([A-Z][a-z]+) (say) (.+?)\.',
    r'([A-Z][a-z]+) (says) (.+?)\.']

pattern = r'([A-Z][a-z]+ [A-Z][a-z]+) (said) (.+?)\.'

# TODO: get literal quotes as "content" [said] Mr. X.

def get_word_counts(fname):
    """
    Given a filename with plain text,
    it returns a Counter object that
    tells how many times each word appeared
    in the text. This text is useful to
    identify low-frequency (and potentially
    interesting) words.
    """
    counter = Counter()
    with open(fname) as fin:
        for line in fin:
            line = line.strip()
            words = [w.lower() for w in line.split()]
            counter.update(words)
    return counter

word_counts = get_word_counts(fname)
subj_counts = Counter()
with open('bbc.txt') as fin:
    for line in fin:
        occs = re.findall(pattern, line)
        if occs:
            agent = occs[0][0]
            rel = occs[0][1]
            subject = occs[0][2]
            words = [w.lower() for w in subject.split()]
            words_freqs = [(w, word_counts[w]) for w in words]
            words_freqs.sort(key=lambda x: x[1])
            keywords = [k for (k, f) in words_freqs][:3]
            subj_counts.update([tuple(keywords)])
            print('Agent: {0}\nRel: {1}\nSubj: {2}\nSubj (raw): {3}'.format(
                agent, rel, keywords, subject))
            print('------------------')

print('Most popular subjects: {0}'.format(
    subj_counts.most_common(5)))
