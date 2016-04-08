<?php

declare (strict_types = 1);

namespace Phpml\Exception;

class InvalidArgumentException extends \Exception
{
    /**
     * @return InvalidArgumentException
     */
    public static function sizeNotMatch()
    {
        return new self('Size of given arguments not match');
    }

    /**
     * @param $name
     *
     * @return InvalidArgumentException
     */
    public static function percentNotInRange($name)
    {
        return new self(sprintf('%s must be between 0.0 and 1.0', $name));
    }
}