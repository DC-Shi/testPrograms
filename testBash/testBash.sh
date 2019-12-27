#!/bin/bash

function haha {
  return $1
}


( haha 1 && echo return0 )||(echo non-zero)
( haha 0 && echo return0 )||(echo non-zero)
