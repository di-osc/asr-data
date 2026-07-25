# Embedded Chinese TN resources

Place the Chinese text-normalization FST files in this directory before building:

```text
assets/wetext/
├── tagger.fst
├── verbalizer.fst
├── verbalizer_remove_erhua.fst
├── traditional_to_simple.fst
├── full_to_half.fst
├── remove_interjections.fst
└── remove_puncts.fst
```

The build embeds all resources into the Rust library and Python extension. They are loaded
directly from memory and are not extracted at runtime. The default TN policy uses `tagger.fst` and
`verbalizer.fst`; callers can independently opt into traditional-to-simplified conversion, the
remove-erhua verbalizer, full-width conversion, interjection removal, and punctuation removal.

The files are derived from the WeText text-normalization resources. Keep their applicable license
and attribution alongside redistributed binaries.
