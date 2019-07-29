<?php

declare(strict_types=1);

namespace Phpml\NeuralNetwork;

use Phpml\Exception\BadNeuralNetworkStructureException;
use Phpml\Exception\InvalidArgumentException;
use Phpml\NeuralNetwork\Node\Neuron;

class Layer
{
    /**
     * @var Node[]
     */
    private $nodes = [];

    /**
     * @throws InvalidArgumentException
     */
    public function __construct(int $nodesNumber = 0, string $nodeClass = Neuron::class, ?ActivationFunction $activationFunction = null)
    {
        if (!in_array(Node::class, class_implements($nodeClass), true)) {
            throw new InvalidArgumentException('Layer node class must implement Node interface');
        }

        for ($i = 0; $i < $nodesNumber; ++$i) {
            $this->nodes[] = $this->createNode($nodeClass, $activationFunction);
        }
    }

    public function addNode(Node $node): void
    {
        $this->nodes[] = $node;
    }

    /**
     * @return Node[]
     */
    public function getNodes(): array
    {
        return $this->nodes;
    }

    public function getTrainedCharacteristics(): array
    {
        $result = [];
        foreach ($this->nodes as $node) {
            if ($node instanceof Neuron) {
                $result[] = $node->getTrainedCharacteristics();
            }
        }

        return $result;
    }

    public function setTrainedCharacteristics(array $characteristics): void
    {
        // iterate over the node instances
        $iNode = -1;
        for ($i = 0; $i < count($this->nodes); $i++) {
            $node = $this->nodes[$i];
            if ($node instanceof Neuron) {
                $iNode ++;

                if (count($characteristics) < $iNode + 1) {
                    throw new BadNeuralNetworkStructureException();
                }

                $node->setTrainedCharacteristics($characteristics[$iNode]);
            }
        }
    }

    private function createNode(string $nodeClass, ?ActivationFunction $activationFunction = null): Node
    {
        if ($nodeClass === Neuron::class) {
            return new Neuron($activationFunction);
        }

        return new $nodeClass();
    }
}
