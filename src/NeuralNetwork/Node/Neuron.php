<?php

declare(strict_types=1);

namespace Phpml\NeuralNetwork\Node;

use Phpml\Exception\BadNeuralNetworkStructureException;
use Phpml\NeuralNetwork\ActivationFunction;
use Phpml\NeuralNetwork\ActivationFunction\Sigmoid;
use Phpml\NeuralNetwork\Node;
use Phpml\NeuralNetwork\Node\Neuron\Synapse;

class Neuron implements Node
{
    /**
     * @var Synapse[]
     */
    protected $synapses = [];

    /**
     * @var ActivationFunction
     */
    protected $activationFunction;

    /**
     * @var float
     */
    protected $output = 0.0;

    /**
     * @var float
     */
    protected $z = 0.0;

    public function __construct(?ActivationFunction $activationFunction = null)
    {
        $this->activationFunction = $activationFunction ?: new Sigmoid();
    }

    public function addSynapse(Synapse $synapse): void
    {
        $this->synapses[] = $synapse;
    }

    /**
     * @return Synapse[]
     */
    public function getSynapses(): array
    {
        return $this->synapses;
    }

    public function getOutput(): float
    {
        if ($this->output === 0.0) {
            $this->z = 0;
            foreach ($this->synapses as $synapse) {
                $this->z += $synapse->getOutput();
            }

            $this->output = $this->activationFunction->compute($this->z);
        }

        return $this->output;
    }

    public function getDerivative(): float
    {
        return $this->activationFunction->differentiate($this->z, $this->output);
    }

    public function reset(): void
    {
        $this->output = 0.0;
        $this->z = 0.0;
    }

    public function getTrainedCharacteristics(): array
    {
        $result = [];
        foreach ($this->synapses as $synapse) {
            $result[] = $synapse->getWeight();
        }

        return $result;
    }

    public function setTrainedCharacteristics(array $characteristics): void
    {
        // length of weights should equals number of synapses
        if (count($characteristics) != count($this->synapses)) {
            throw new BadNeuralNetworkStructureException();
        }

        for ($i = 0; $i < count($characteristics); $i++) {
            $this->synapses[$i]->setWeight($characteristics[$i]);
        }
    }
}
