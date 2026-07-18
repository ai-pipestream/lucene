# Lucene preview build (unofficial, experimental)

This branch and the `ai.pipestream:lucene-*` artifacts built from it are an
**experimental, unsupported preview**, built by adding unmerged development
work to [Apache Lucene](https://lucene.apache.org/) ahead of its review and
release. **This is not an Apache Software Foundation release**, is not
endorsed by the ASF, and nothing in it is promised for any future Lucene
version. Apache Lucene, Lucene, and Apache are trademarks of the Apache
Software Foundation.

What this exists for: downstream projects that want to build against the
in-review shared-floor kNN collection classes
(`org.apache.lucene.sandbox.search.knn`, apache/lucene#16357) without
pressuring the upstream review queue. When the corresponding feature ships in
an official Apache Lucene release, switch your dependency to the official
`org.apache.lucene` coordinates; the Java package names are identical by
design, so migration is a coordinate swap.

Exactly which refs a build contains is recorded in
`PIPESTREAM-PROVENANCE.txt` at the repository root: the upstream main commit,
the build date, and every feature commit on top of it. Artifacts are
published exclusively as Maven snapshots (`11.0.0-experimental-SNAPSHOT`):
snapshot semantics are the point, nothing built here is ever a release, and
the manifest is what says which snapshot you actually have.

Licensing: Apache License 2.0, unchanged. The upstream `LICENSE` and `NOTICE`
files are retained as-is; this file and the provenance manifest constitute
the statement of modification required by section 4 of the license. Report
problems with this preview to the ai-pipestream repository, never to the
Apache Lucene project; upstream owns none of this build.

Build group differs from upstream deliberately: `ai.pipestream` instead of
`org.apache.lucene`, so this line can never shadow or be shadowed by official
Lucene artifacts on a classpath or in a repository.
