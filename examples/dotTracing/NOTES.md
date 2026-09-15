# MODEL changelog
added competition amoung the motor layer

rewarding: made the change in distance only based on the agent's movement

added an evaluation number that will hopefully reflect agent performance



# PARAM change history

Increase threashold on the place cell population to be more selective (this layer fires too much)
-45 -> -35 -> -25

Simplified the recurrent connection + decrease magnitude of connections in reservoir

Tuned the magnitude of the recurrent connection scaler (removed for now)

Removed the scaling value on the ac_mc connection

Increse magnitude of pc_ac connection scaler due to low/high spiking of ac layer
from as low as 5 -> 25 -> 80 -> 160 -> 120 (havent tested 120 yet)

Tuned the threashold of the association layer to allow for more sparse firing

Added a bounds on the possible values of the learned weights (0 to 3) subject to change

PC_A conenction weights are decreased to make the agent's place cell firing have
less of an impace on the overall decision

Recurrent is just 50% pos and 50% neg- fixed

Increased the sparsity of the ac_mc layer to prevent any single ac neuron
to over-fire the mc layer

Decreased the normalization of the recurrent connection 
avg sum of column -> 60 -> 30
This is done to hopefully encourage more competition and sparsity

Changed the starting prime number in grid cell back to 3


# TODO

EVOL: add the scaling factors of each weights AND add the ac_mc connection

MAYBE: include more improvements on the fitness scoring


add more sparsity as time goes on - unsure


Improve upon place cell, I should be able to tell where the target+agent are
just by looking at the place cell spiking


# ISSUES/documentation

8/31
Spiking of the ac and mc layer needs to be tuned.

Low competition in mc layer

idk what to do about the decaying tail, probably my current method of input is good


9/1
Place cells randomly begin spiking a lot more than usual, which causes vertical lines
of firing in the ac and mc layer. This always seems to force the motor layer into a 
horizontal line

9/9
Vertical lines have disappeared, 




