"""
animal_classes.py
This Module contains the List of the Animal Classes
"""
# Define animal classes and their respective indices for Animal Classification
animal_classes = [
    "acinonyx-jubatus",
    "aethia-cristatella",
    "african_elephant",
    "agalychnis-callidryas",
    "agkistrodon-contortrix",
    "ailuropoda-melanoleuca",
    "ailurus-fulgens",
    "alces-alces",
    "alpaca",
    "american_bison",
    "anas-platyrhynchos",
    "ankylosaurus-magniventris",
    "anteater",
    "antelope",
    "apis-mellifera",
    "aptenodytes-forsteri",
    "aquila-chrysaetos",
    "ara-macao",
    "architeuthis-dux",
    "arctic_fox",
    "ardea-herodias",
    "armadillo",
    "baboon",
    "badger",
    "balaenoptera-musculus",
    "bat",
    "bear",
    "bee",
    "beetle",
    "betta-splendens",
    "Bird",
    "bison",
    "blue_whale",
    "boar",
    "bos-gaurus",
    "bos-taurus",
    "bradypus-variegatus",
    "branta-canadensis",
    "brown_bear",
    "butterfly",
    "camel",
    "canis-lupus",
    "canis-lupus-familiaris",
    "carcharodon-carcharias",
    "cardinalis-cardinalis",
    "cat",
    "caterpillar",
    "cathartes-aura",
    "centrochelys-sulcata",
    "centruroides-vittatus",
    "ceratitis-capitata",
    "ceratotherium-simum",
    "chelonia-mydas",
    "chimpanzee",
    "chrysemys-picta",
    "circus-hudsonius",
    "Clams",
    "cockroach",
    "codium-fragile",
    "coelacanthiformes",
    "colaptes-auratus",
    "connochaetes-gnou",
    "Corals",
    "correlophus-ciliatus",
    "cow",
    "coyote",
    "Crabs",
    "crocodylus-niloticus",
    "crotalus-atrox",
    "crotophaga-sulcirostris",
    "crow",
    "cryptoprocta-ferox",
    "cyanocitta-cristata",
    "danaus-plexippus",
    "dasypus-novemcinctus",
    "deer",
    "delphinapterus-leucas",
    "dendrobatidae",
    "dermochelys-coriacea",
    "desmodus-rotundus",
    "diplodocus",
    "dog",
    "dolphin",
    "donkey",
    "dragonfly",
    "duck",
    "dugong-dugon",
    "eagle",
    "Eel",
    "eidolon-helvum",
    "elephant",
    "enhydra-lutris",
    "enteroctopus-dofleini",
    "equus-caballus",
    "equus-quagga",
    "eudocimus-albus",
    "eunectes-murinus",
    "falco-peregrinus",
    "felis-catus",
    "Fish",
    "flamingo",
    "fly",
    "formicidae",
    "fox",
    "gallus-gallus-domesticus",
    "gavialis-gangeticus",
    "geococcyx-californianus",
    "giraffa-camelopardalis",
    "Giraffe",
    "goat",
    "goldfish",
    "goose",
    "gorilla",
    "grasshopper",
    "groundhog",
    "haliaeetus-leucocephalus",
    "hamster",
    "hapalochlaena-maculosa",
    "hare",
    "hedgehog",
    "heloderma-suspectum",
    "heterocera",
    "highland_cattle",
    "hippopotamus",
    "homo-sapiens",
    "hornbill",
    "horse",
    "hummingbird",
    "hydrurga-leptonyx",
    "hyena",
    "icterus-galbula",
    "icterus-gularis",
    "icterus-spurius",
    "iguana-iguana",
    "iguanodon-bernissartensis",
    "inia-geoffrensis",
    "jackal",
    "Jelly Fish",
    "kangaroo",
    "koala",
    "ladybugs",
    "lampropeltis-triangulum",
    "lemur-catta",
    "leopard",
    "lepus-americanus",
    "lion",
    "lizard",
    "lobster",
    "loxodonta-africana",
    "macropus-giganteus",
    "malayopython-reticulatus",
    "mammuthus-primigeniu",
    "manatee",
    "martes-americana",
    "megaptera-novaeangliae",
    "melanerpes-carolinus",
    "mellisuga-helenae",
    "mergus-serrator",
    "mimus-polyglottos",
    "mongoose",
    "monodon-monoceros",
    "mosquito",
    "moth",
    "mountain_goat",
    "mouse",
    "musca-domestica",
    "Nudibranchs",
    "octopus",
    "odobenus-rosmarus",
    "okapia-johnstoni",
    "ophiophagus-hannah",
    "opossum",
    "orangutan",
    "orcinus-orca",
    "ornithorhynchus-anatinus",
    "otter",
    "ovis-aries",
    "ovis-canadensis",
    "owl",
    "ox",
    "oyster",
    "panda",
    "panthera-leo",
    "panthera-onca",
    "panthera-pardus",
    "panthera-tigris",
    "pantherophis-alleghaniensis",
    "pantherophis-guttatus",
    "papilio-glaucus",
    "parrot",
    "passerina-ciris",
    "pavo-cristatus",
    "pelecaniformes",
    "penguin",
    "periplaneta-americana",
    "phascolarctos-cinereus",
    "phoebetria-fusca",
    "phoenicopterus-ruber",
    "phyllobates-terribilis",
    "physalia-physalis",
    "physeter-macrocephalus",
    "pig",
    "pigeon",
    "poecile-atricapillus",
    "polar_bear",
    "pongo-abelii",
    "porcupine",
    "possum",
    "procyon-lotor",
    "pteranodon-longiceps",
    "pterois-mombasae",
    "pterois-volitans",
    "Puffers",
    "puma-concolor",
    "raccoon",
    "rat",
    "rattus-rattus",
    "red_panda",
    "reindeer",
    "rhinoceros",
    "rusa-unicolor",
    "salmo-salar",
    "sandpiper",
    "sciurus-carolinensis",
    "Sea Rays",
    "Sea Urchins",
    "seahorse",
    "seal",
    "sea_lion",
    "Sharks",
    "sheep",
    "Shrimp",
    "smilodon-populator",
    "snake",
    "snow_leopard",
    "sparrow",
    "spheniscus-demersus",
    "sphyrna-mokarran",
    "spinosaurus-aegyptiacus",
    "squid",
    "squirrel",
    "starfish",
    "stegosaurus-stenops",
    "struthio-camelus",
    "sugar_glider",
    "swan",
    "tapir",
    "tapirus",
    "tarsius-pumilus",
    "taurotragus-oryx",
    "telmatobufo-bullocki",
    "thryothorus-ludovicianus",
    "tiger",
    "triceratops-horridus",
    "trilobita",
    "turdus-migratorius",
    "turkey",
    "tursiops-truncatus",
    "Turtle_Tortoise",
    "tyrannosaurus-rex",
    "tyrannus-tyrannus",
    "ursus-arctos-horribilis",
    "ursus-maritimus",
    "vampire_bat",
    "varanus-komodoensis",
    "vicuna",
    "vulpes-vulpes",
    "vultur-gryphus",
    "walrus",
    "warthog",
    "water_buffalo",
    "weasel",
    "whale",
    "wildebeest",
    "wolf",
    "wombat",
    "woodpecker",
    "yak",
    "zebra",
]
