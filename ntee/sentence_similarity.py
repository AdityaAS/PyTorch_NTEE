import click
import os
import re
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from ntee.utils.model_reader import ModelReader
from ntee.utils.model_reader import PyTorchModelReader

def eval_sts(model_file, vocab_file, dataset_dir, modelname, out_file):
    reader = PyTorchModelReader(model_file, vocab_file, modelname, cpuflag=True)
    fout = open(out_file, 'w')
    for file_name in sorted(os.listdir(dataset_dir)):
        match_obj = re.match(r'^STS\.input\.(.*)\.txt', file_name)
        if not match_obj:
            continue

        name = match_obj.group(1)
        predicted = []
        correct = []

        with open(os.path.join(dataset_dir, file_name)) as input_file:
            with open(os.path.join(dataset_dir, 'STS.gs.' + name + '.txt')) as gs_file:
                for (line, score) in zip(input_file, gs_file):
                    score = score.rstrip()
                    if not score:
                        continue
                    score = float(score)

                    (sent1, sent2) = line.rstrip().split('\t')
                    correct.append(score)

                    vec1 = reader.get_text_vector(sent1).numpy()
                    vec2 = reader.get_text_vector(sent2).numpy()
                    predicted.append(1.0 - cosine(vec1, vec2))
                    
        click.echo('%s: %.4f (pearson) %.4f (spearman)' % (
            name, pearsonr(correct, predicted)[0], spearmanr(correct, predicted)[0] ))
        fout.write('%s: %.4f (pearson) %.4f (spearman)' % (
            name, pearsonr(correct, predicted)[0], spearmanr(correct, predicted)[0] ))
        fout.write('\n')

    fout.close()
def eval_sick(model_file, vocab_file, dataset_file, modelname, out_file):
    reader = PyTorchModelReader(model_file, vocab_file, modelname, cpuflag=True)
    fout = open(out_file, 'w')

    predicted = []
    correct = []

    with open(dataset_file, 'r') as fin:
        for (n, line) in enumerate(fin):
            if n == 0:
                continue

            data = line.rstrip().split('\t')
            sent1 = data[1]
            sent2 = data[2]
            score = float(data[4])
            fold = data[11]

            if fold == 'TRIAL':
                continue

            correct.append(float(score))
            vec1 = reader.get_text_vector(sent1).numpy()
            vec2 = reader.get_text_vector(sent2).numpy()
            predicted.append(1.0 - cosine(vec1, vec2))

    pc = pearsonr(correct, predicted)[0]
    sc = spearmanr(correct, predicted)[0]
    fout.write('%.4f (pearson) %.4f (spearman)' % (pc, sc))
    click.echo('%.4f (pearson) %.4f (spearman)' % (pc, sc))
    fout.close()
    return pc, sc