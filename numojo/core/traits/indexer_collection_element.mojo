trait IndexerCollectionElement(Indexer, CollectionElement):
    """The IndexerCollectionElement trait denotes a trait composition
    of the `Indexer` and `CollectionElement` traits.

    This is useful to have as a named entity since Mojo does not
    currently support anonymous trait compositions to constrain
    on `Indexer & CollectionElement` in the parameter.
    """

    pass
