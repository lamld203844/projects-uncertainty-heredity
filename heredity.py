import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                            False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1

    # Loop over all person
    for person in people:
        
        # GEN PROBABILITY
        gen_prob = 1
        # init number of gen(es), = 0 by default for using later
        copies = 0
        # If have not parent
        mother = people[person]['mother']
        father = people[person]['father']
        if mother is None and father is None:
            if person in two_genes:
                copies = 2
            elif person in one_gene:
                copies = 1
            gen_prob = PROBS["gene"][copies]
        # Else, having parent -> compute based on conditional on what genes their parents have.
        else:
            # Compute probability dict reference of getting gen ability from each parent
            parent_dict = {mother: 1, father: 1}
            # Consider each mother/father
            for parent in parent_dict:
                # Case having 2 gene -> chance of passing on it = (1-P(mutation))
                if parent in two_genes:
                    parent_dict[parent] *= 1 - PROBS["mutation"]
                # ??? Case having 1 gene -> passing copy gen without mutation + passing non-copy gen with mutation
                elif parent in one_gene:
                    parent_dict[parent] *= 0.5 * (1 - PROBS["mutation"]) #+ 0.5 * PROBS["mutation"]
                # Case having 0 gene -> only passing via mutating
                else:
                    parent_dict[parent] *= PROBS["mutation"]

            # Calculate gen_prob based on parent_prob dict
            # son has 2 gen = P(1_from_mother) * P(1_from_father)
            if person in two_genes:
                copies = 2
                gen_prob = parent_dict[mother] * parent_dict[father]
            # son has 1 gen = P(1_from_mother) * P(0_from_father) + vice versa
            elif person in one_gene:
                copies = 1
                # 1 from mother, 0 from father
                prob_1 = parent_dict[mother] * (1 - parent_dict[father]) 
                # 0 from mother, 1 from father 
                prob_2 = (1 - parent_dict[mother]) * parent_dict[father]
                gen_prob *= (prob_1 + prob_2)
            # son has 0 gen, neither from father and mother
            else:
                gen_prob = (1 - parent_dict[mother]) * (1 - parent_dict[father]) 

        # TRAIT PROBABILITY base on number of gen(es)
        trait_prob = 1
        if person in have_trait:
            trait_prob *= PROBS["trait"][copies][True]
        else:
            trait_prob *= PROBS["trait"][copies][False]
        
        # update joint probability
        joint_prob *= gen_prob * trait_prob

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # update person is in one gene 
    for person in one_gene:
        probabilities[person]['gene'][1] += p
    
    # update person is in two gene
    for person in two_genes:
        probabilities[person]['gene'][2] += p

    # update person have trait
    for person in have_trait:
        probabilities[person]['trait'][True] += p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Loop over each person
    for person in probabilities:
        # normalize by finding normalizing constant factor alpha
        for attribute in probabilities[person]:
            # finding normalizing constant factor alpha
            sum = 0
            for type in probabilities[person][attribute]:
                sum += probabilities[person][attribute][type]

            if sum == 0:
                sum = 1
            alpha = 1/sum

            # update gene/trait prob distribution
            for type in probabilities[person][attribute]:
                probabilities[person][attribute][type] *= alpha

if __name__ == "__main__":
    main()
