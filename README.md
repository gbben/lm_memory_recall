# Studying memory recall in Language Models
An (ambitious?) project to investiage memory recall in Language Models.

## First approach -- follow the path of something like Statistical Parametric Mappings (SPM):


### SPM experimental design:
Experimental Design:

Participants undergo fMRI scanning while performing memory tasks
The experiment would use something like a "block design" or "event-related design" where:

* Memory recall periods are alternated with control/baseline periods
* Different types of memories might be tested (e.g., autobiographical vs. semantic)
* Timing of stimuli presentation and responses would be precisely controlled



Rather than simply asking the same questions repeatedly in different ways more sophisticated paradigms are employed:

#### Memory Types Comparison:

* Autobiographical memories ("Tell me about your last birthday")
* Semantic memories ("What's the capital of France?")
* Recent vs. remote memories
* True vs. false memories


#### Control Tasks:

* Rest periods
* Non-memory cognitive tasks
* Novel information processing


### Potential outline of (experimental) design for LMs:



#### Question Types and Controls:

##### Memory/Fact Recall Questions:

* Direct fact queries: "What is the capital of France?"
* Paraphrased fact queries: "Which city serves as France's capital?"
* Context-embedded queries: "If you were visiting France's seat of government, which city would you be in?"
* Time-varied queries: "What was France's capital in 1950?"

##### Control Questions (matched for complexity but not requiring fact recall):

* Logic questions: "If A is greater than B, and B is greater than C, is A greater than C?"
* Pattern completion: "Complete the sequence: 2, 4, 6, ..."
* Linguistic tasks: "Is this sentence grammatically correct?"
* Novel inference: "If cars could fly, what would roads be used for?"


#### Experimental Design:

##### A. Block Design:

* Run blocks of memory questions followed by control questions
* Alternate between different types of memory questions
* Include rest periods (simple text processing tasks)

##### B. Controlled Variables:

* Match question length
* Control for token count
* Maintain similar grammatical complexity
* Balance positive/negative answers


# First approach in the code:

* Track activations across different layers
* Compare activation patterns between memory and control tasks
* Identify statistically significant differences in activation
* Map which neurons/attention heads are most active during memory recall