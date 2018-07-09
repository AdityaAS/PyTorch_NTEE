#!/usr/bin/python
#-*- coding: utf-8 -*-

import bz2
import logging
import re
from gensim.corpora import wikicorpus

logger = logging.getLogger(__name__)

DEFAULT_IGNORED_NS = (
    'wikipedia:', 'file:', 'portal:', 'template:', 'mediawiki:', 'user:',
    'help:', 'book:', 'draft:'
)

REDIRECT_REGEXP = re.compile(
    r"(?:\#|＃)(?:REDIRECT|転送)[:\s]*(?:\[\[(.*)\]\]|(.*))", re.IGNORECASE
)

class WikiPage(object):
    __slots__ = ('title', 'language', 'wiki_text')

    def __init__(self, title, language, wiki_text):
        self.title = title
        self.language = language
        self.wiki_text = wiki_text

    def __repr__(self):
        return '<WikiPage %s>' % (self.title.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.title, self.language, self.wiki_text))

    @property
    def is_redirect(self):
        return bool(self.redirect)

    @property
    def redirect(self):
        red_match_obj = REDIRECT_REGEXP.match(self.wiki_text)
        if not red_match_obj:
            return None

        if red_match_obj.group(1):
            dest = red_match_obj.group(1)
        else:
            dest = red_match_obj.group(2)

        if dest:
            return self._normalize_title(dest)
        else:
            return None

    def _normalize_title(self, title):
        return title.lower().replace('_', ' ')

class WikiDumpReader(object):
    def __init__(self, dump_file, ignored_ns=DEFAULT_IGNORED_NS):
        self._dump_file = dump_file
        self._ignored_ns = ignored_ns

        with bz2.BZ2File(self._dump_file) as f:
            self._language = re.search(r'xml:lang="(.*)"', f.readline().decode('utf-8')).group(1)

    @property
    def dump_file(self):
        return self._dump_file

    @property
    def language(self):
        return self._language

    def __iter__(self):
        with bz2.BZ2File(self._dump_file) as f:
            c = 0
            for (title, wiki_text, wiki_id) in wikicorpus.extract_pages(f):
                if any(
                    [title.lower().startswith(ns) for ns in self._ignored_ns]
                ):
                    continue
                c += 1

                yield WikiPage(
                    str(title.lower()), self._language, str(wiki_text)
                )

                if c % 10000 == 0:
                    logger.info('Processed: %d', c)
