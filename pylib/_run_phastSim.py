# adapted from https://github.com/NicolaDM/phastSim/blob/653ce26c0806a9904a85252c50d05ea59df7e428/bin/phastSim
import contextlib
from functools import wraps
from io import StringIO
import itertools as it
import pathlib
import sys
import tempfile

from ete3 import Tree
from hstrat import _auxiliary_lib as hstrat_aux
import pandas as pd
import phastSim.phastSim as phastSim
import polars as pl


def _with_work_dir(**tempdir_kwargs):
    """
    Decorator that provides a temporary working directory to the decorated function.
    The directory is automatically cleaned up after the function exits.

    The decorated function must accept the temp directory path as its first argument.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with tempfile.TemporaryDirectory(**tempdir_kwargs) as tmpdir:
                work_dir = pathlib.Path(tmpdir)
                return fn(*args, **kwargs, work_dir=work_dir)

        return wrapper

    return decorator


class FilteredStream:
    def __init__(self, original_stream, prefix):
        self.original_stream = original_stream
        self.prefix = prefix
        self._buffer = ""

    def write(self, data):
        # Buffer incoming data and process full lines
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            # Only write lines that start with the prefix
            if line.startswith(self.prefix):
                self.original_stream.write(line + "\n")

    def flush(self):
        # Flush any remaining buffered data if it matches
        if self._buffer and self._buffer.startswith(self.prefix):
            self.original_stream.write(self._buffer)
        self._buffer = ""
        self.original_stream.flush()


@contextlib.contextmanager
def filtered_output(prefix):
    """Context manager that only lets lines starting with `prefix` pass
    through."""
    # Wrap stdout and stderr
    f_out = FilteredStream(sys.stdout, prefix)
    f_err = FilteredStream(sys.stderr, prefix)
    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
        yield


@filtered_output(prefix="!!!")
@_with_work_dir(suffix="_phastSim")
def _do_run_phastSim(
    ancestral_sequence: str,
    phylo_newick: str,
    *,
    work_dir: pathlib.Path,
) -> pl.DataFrame:
    """Shim function to run phastSim without subprocess."""

    # write ancestral sequence to tempdir
    ancestral_sequence_path = work_dir / "ancestral_sequence.fasta"
    ancestral_sequence_path.write_text(
        f">ancestral_sequence\n{ancestral_sequence}",
    )

    shim_args = [
        "--reference",
        str(ancestral_sequence_path),
        "--outpath",
        str(work_dir),
        "--scale",
        "2.74e-6",
    ]

    # setup the argument parser and read the arguments from command line
    parser = phastSim.setup_argument_parser()
    args = parser.parse_args(args=shim_args)

    # instantiate a phastSim run. This class holds all arguments and constants,
    # which can be easily called as e.g. sim_run.args.path or
    # sim_run.const.alleles
    with hstrat_aux.log_context_duration("phastSimRun", print):
        sim_run = phastSim.phastSimRun(args=args)
    # np.random.seed(args.seed)
    hierarchy = not args.noHierarchy

    # initialise the root genome. Reads either from file or creates a genome in
    # codon or nucleotide mode
    with hstrat_aux.log_context_duration("init_...", print):
        ref, refList = sim_run.init_rootGenome()

        # set up substitution rates
        mutMatrix = sim_run.init_substitution_rates()

        # set up gamma rates
        if args.mutationsTSVinput is not None:
            gammaRates, preMutationsBranches = sim_run.init_gamma_rates()
        else:
            gammaRates = sim_run.init_gamma_rates()
            preMutationsBranches = {}

        # set up hypermutation rates
        hyperCategories = sim_run.init_hypermutation_rates()

        # set up codon substitution model
        if sim_run.args.codon:
            omegas = sim_run.init_codon_substitution_model()
        else:
            omegas = None

        # can only use indels in heirarchy mode
        if not hierarchy and sim_run.args.indels:
            print(
                "indels cannot be used with the -- noHierarchy option, exiting."
            )
            exit()

        # set up arguments for indel model
        (
            insertion_rates,
            insertion_lengths,
            insertion_frequencies,
            deletion_rates,
            deletion_lengths,
        ) = ([], [], [], [], [])

        if sim_run.args.indels:

            insertion_rates, deletion_rates = sim_run.init_indel_rates()
            insertion_lengths, deletion_lengths = sim_run.init_indel_lengths()
            insertion_frequencies = sim_run.init_insertion_frequencies(ref)

    # Loads a tree structure from a newick string in ETE3. The returned
    # variable t is the root node for the tree.
    if args.mutationsTSVinput is not None:
        if args.eteFormat != 1:
            print(
                "WARNING: ete3 parsing format %s is inappropriate for "
                "use with mutationsTSVinput. Setting to parsing mode 1"
                % args.eteFormat
            )
            args.eteFormat = 1

    with hstrat_aux.log_context_duration("ete3Tree", print):
        t = Tree(phylo_newick, format=args.eteFormat)

    # save information about the categories of each site on a file
    if args.createInfo:
        infoFile = open(args.outpath + args.outputFile + ".info", "w")
        if args.codon:
            headerString = (
                "pos\t"
                + "omega\t"
                + "cat1\t"
                + "hyperCat1\t"
                + "hyperAlleleFrom1\t"
                + "hyperAlleleTo1\t"
                + "cat2\t"
                + "hyperCat2\t"
                + "hyperAlleleFrom2\t"
                + "hyperAlleleTo2\t"
                + "cat3\t"
                + "hyperCat3\t"
                + "hyperAlleleFrom3\t"
                + "hyperAlleleTo3"
            )
        else:
            headerString = (
                "pos\t"
                + "cat\t"
                + "hyperCat\t"
                + "hyperAlleleFrom\t"
                + "hyperAlleleTo"
            )
        if args.indels:
            headerString += "\t" + "insertionRate\t" + "deletionRate"
            headerString = "insertionPos\t" + headerString

        headerString += "\n"
        infoFile.write(headerString)
    else:
        infoFile = None

    # Hierarchical approach (DIVIDE ET IMPERA ALONG THE GENOME),
    # WHEN THE RATES ARE UPDATED, UPDATE ONLY RATE OF THE SITE AND OF ALL THE
    # NODES ON TOP OF THE HYRARCHY. THAT IS, DEFINE A tree STRUCTURE, WITH
    # TERMINAL NODES BEING GENOME LOCI AND WITH INTERNAL NODES BING MERGING
    # GROUPS OF LOCI CONTAINING INFORMATION ABOUT THEIR CUMULATIVE RATES. THIS
    # WAY UPDATING A MUTATION EVENT REQUIRES COST LOGARITHMIC IN GENOME SIZE
    # EVEN IF EVERY SITE HAS A DIFFERENT RATE.
    if hierarchy:
        # CODONS: don't create all the matrices from the start (has too large a
        # memory and time preparation cost). instead, initialize only the rates
        # from the reference allele (only 9 rates are needed), and store them in
        # a dictionary at level 0 terminal nodes, and when new codons at a
        # position are reached, extend the dictionary and calculate these new
        # rates. Most positions will have only a few codons explored.
        with hstrat_aux.log_context_duration("GenomeTree_hierarchical", print):
            # instantiate a GenomeTree with all needed rates and categories
            genome_tree = phastSim.GenomeTree_hierarchical(
                nCodons=sim_run.nCodons,
                codon=sim_run.args.codon,
                ref=ref,
                gammaRates=gammaRates,
                omegas=omegas,
                mutMatrix=mutMatrix,
                hyperCategories=hyperCategories,
                hyperMutRates=sim_run.args.hyperMutRates,
                indels=sim_run.args.indels,
                insertionRate=insertion_rates,
                insertionLength=insertion_lengths,
                insertionFrequencies=insertion_frequencies,
                deletionRate=deletion_rates,
                deletionLength=deletion_lengths,
                scale=sim_run.args.scale,
                infoFile=infoFile,
                verbose=sim_run.args.verbose,
                noNorm=sim_run.args.noNormalization,
                mutationsTSVinput=sim_run.args.mutationsTSVinput,
            )

            # populate the GenomeTree
            genome_tree.populate_genome_tree()

            # check start and stop codons and normalize all rates
            if genome_tree.codon:
                genome_tree.check_start_stop_codons()

            genome_tree.normalize_rates()

        # NOW DO THE ACTUAL SIMULATIONS. DEFINE TEMPORARY STRUCTURE ON TOP OF
        # THE CONSTANT REFERENCE GENOME TREE. define a multi-layered tree; we
        # start the simulations with a genome tree. as we move down the
        # phylogenetic tree, new layers are added below the starting tree. Nodes
        # to layers below link to nodes above, or to nodes on the same layer,
        # but never to nodes in the layer below. while traversing the tree, as
        # we move up gain from a node back to its parent (so that we can move to
        # siblings etc), the nodes in layers below the current one are simply
        # "forgotten" (in C they could be de-allocated, but the task here is
        # left to python automation).
        with hstrat_aux.log_context_duration(
            "mutateBranchETEhierarchy", print
        ):
            genome_tree.mutateBranchETEhierarchy(
                t,
                genome_tree.genomeRoot,
                1,
                sim_run.args.createNewick,
                preMutationsBranches,
            )

    # use simpler approach that collates same rates along the genome - less
    # #efficient with more complex models.
    else:
        with hstrat_aux.log_context_duration("GenomeTree_vanilla", print):
            # instantiate a genome tree for the non hierarchical case
            genome_tree = phastSim.GenomeTree_vanilla(
                nCat=sim_run.nCat,
                ref=ref,
                mutMatrix=mutMatrix,
                categories=list(sim_run.categories),
                categoryRates=sim_run.args.categoryRates,
                hyperMutRates=sim_run.args.hyperMutRates,
                hyperCategories=list(hyperCategories),
                infoFile=infoFile,
                verbose=sim_run.args.verbose,
            )

            # prepare the associated lists and mutation rate matrices
            genome_tree.prepare_genome()

            # normalize the mutation rates
            genome_tree.normalize_rates(scale=sim_run.args.scale)

        # Run sequence evolution simulation along tree
        with hstrat_aux.log_context_duration("mutateBranchETE", print):
            genome_tree.mutateBranchETE(
                t,
                genome_tree.muts,
                genome_tree.totAlleles,
                genome_tree.totMut,
                genome_tree.extras,
                sim_run.args.createNewick,
            )

    # depending on the type of genome_tree, this automatically uses the correct
    # version
    with hstrat_aux.log_context_duration("write_genome_short", print):
        genome_tree.write_genome_short(
            tree=t,
            output_path=args.outpath,
            output_file=args.outputFile,
            alternative_output_format=args.alternativeOutputFormat,
        )

    # adapted from
    # https://github.com/NicolaDM/phastSim/blob/653ce26c0806a9904a85252c50d05ea59df7e428/phastSim/phastSim.py#L195
    stringio = StringIO()
    if genome_tree.indels:
        with hstrat_aux.log_context_duration("writeGenomeIndels", print):
            mutDict, insertionDict = {}, {}
            genome_tree.writeGenomeIndels(
                node=t,
                file=stringio,
                mutDict=mutDict,
                insertionDict=insertionDict,
            )

    else:
        with hstrat_aux.log_context_duration("writeGenomeNoIndels", print):
            genome_tree.writeGenomeNoIndels(t, stringio, refList)

    with hstrat_aux.log_context_duration("pl.DataFrame", print):
        lines = stringio.getvalue().splitlines()
        assert len(lines) == 0 or lines[0].startswith(">")
        assert len(lines) == 0 or not lines[1].startswith(">")

        return pl.DataFrame(
            {
                "id": [int(line[1:]) for line in lines[0::2]],
                "sequence": [line for line in lines[1::2]],
            },
        )


def run_phastSim(
    ancestral_sequence: str,
    phylogeny_df: str,
    taxon_label: str = "id",
) -> pd.DataFrame:
    """Shim function to run phastSim without subprocess."""

    # temporarily remove "-" characters
    with hstrat_aux.log_context_duration("remove dashes", print):
        dash_indices = [
            i for i, char in enumerate(ancestral_sequence) if char == "-"
        ]
        ancestral_sequence_ = ancestral_sequence
        ancestral_sequence = ancestral_sequence.replace("-", "")

    # check for whitespace
    if ancestral_sequence != "".join(ancestral_sequence.split()):
        raise ValueError("Ancestral sequence contains whitespace")

    print(f"{len(phylogeny_df)=}, {len(ancestral_sequence)=}")
    with hstrat_aux.log_context_duration("alifestd_as_newick_asexual", print):
        as_newick = hstrat_aux.alifestd_as_newick_asexual(
            phylogeny_df, taxon_label=taxon_label
        )

    with hstrat_aux.log_context_duration("_do_run_phastSim", print):
        res = _do_run_phastSim(
            ancestral_sequence=ancestral_sequence,
            phylo_newick=as_newick,
        )

    # restore "-" characters
    with hstrat_aux.log_context_duration("restore dashes", print):
        segments = [
            pl.col("sequence").str.slice(
                # shift start forward by 1 for every dash removed (i > 0)
                apos - i + int(bool(i)),
                # length is the gap between dashes
                bpos - apos - int(bool(i)),
            )
            for i, (apos, bpos) in enumerate(
                it.pairwise(
                    [
                        0,
                        *dash_indices,
                        len(ancestral_sequence_),
                    ],
                )
            )
        ]
        res = (
            res.lazy()
            .with_columns(sequence=pl.concat_str(segments, separator="-"))
            .collect()
        )

    assert res["sequence"].str.len_chars().unique().item() == len(
        ancestral_sequence_,
    )

    assert len(res) == hstrat_aux.alifestd_count_leaf_nodes(phylogeny_df)
    assert all(res["sequence"].first()[pos] == "-" for pos in dash_indices)

    return res.to_pandas()
