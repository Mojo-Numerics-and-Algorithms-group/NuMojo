# ===----------------------------------------------------------------------=== #
# NuMojo: Indexer collection element trait
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #

"""
Indexer Collection Element (numojo.core.traits.indexer_collection_element).
==========================================================================

Trait composition for indexer collection elements.

Defines trait composition of `Indexer` and `CollectionElement` traits for use
as constraints in generic parameters.

Exports
-------
- `IndexerCollectionElement`: Trait composition type.
"""

comptime IndexerCollectionElement = Indexer & Copyable

# trait IndexerCollectionElement(Copyable, Indexer, Movable):
#     """The IndexerCollectionElement trait denotes a trait composition
#     of the `Indexer` and `CollectionElement` traits.

#     This is useful to have as a named entity since Mojo does not
#     currently support anonymous trait compositions to constrain
#     on `Indexer & CollectionElement` in the parameter.
#     """

#     pass
