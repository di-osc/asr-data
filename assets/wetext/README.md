# Embedded Chinese TN resources

Place the Chinese text-normalization FST files in this directory before building:

```text
assets/wetext/
├── tagger.fst
├── verbalizer.fst
└── verbalizer_remove_erhua.fst
```

The build embeds `tagger.fst` and `verbalizer.fst` into the Rust library and Python extension.
They are loaded directly from memory and are not extracted at runtime. The remove-erhua variant is
kept with the resource set for future policy selection, but the current default TN pipeline does
not embed or use it.

The files are derived from the WeText text-normalization resources. Keep their applicable license
and attribution alongside redistributed binaries.
