{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE ParallelListComp #-}
-- | 'Approx' instances for __dynamic programming__: using a table to
-- represent a function with a finite domain /exactly/.
--
-- While the 'Approx' class is primarily meant for function
-- approximations, it can also be used for exact dynamic
-- programming. This lets us have a single implementation of
-- algorithms like Bellman iteration that can be run in both "exact"
-- and "approximate" modes.
module RL.Approx.Tabular where

import qualified Data.Foldable                           as Foldable
import           Data.HashMap.Strict                      ( (!)
                                                          , HashMap
                                                          )
import qualified Data.HashMap.Strict                     as HashMap
import           Data.Hashable                            ( Hashable )
import qualified Data.Vector                             as V

import           Numeric.LinearAlgebra                    ( R
                                                          , Vector
                                                          )
import qualified Numeric.LinearAlgebra                   as Matrix

import           RL.Approx.Approx                         ( Approx(..) )

import qualified RL.Matrix                               as Matrix
import           RL.Vector                                ( Affine(..) )

-- | An 'Approx' that models a function as a map.
data Tabular a = Tabular
  { mapping :: !(HashMap a R)
  -- ^ Values for each possible input to the function.
  , domain  :: !(V.Vector a)
  -- ^ Every input value this 'Tabular' can handle, in some fixed
  -- order. Should contain the same values as @HashMap.keys mapping@.
  }
  deriving (Show, Eq)

-- | Return a vector with the saved values for every input, in the
-- same order as 'domain'.
toVector :: (Eq a, Hashable a) => Tabular a -> Vector R
toVector dynamic = Matrix.storable $ eval dynamic <$> domain dynamic

-- | Given a domain and a vector of values, build the corresponding
-- 'Tabular'.
fromVector :: (Eq a, Hashable a) => V.Vector a -> Vector R -> Tabular a
fromVector domain vector = Tabular { domain, mapping }
  where mapping = HashMap.fromList $ V.toList domain `zip` Matrix.toList vector

instance (Eq a, Hashable a) => Affine (Tabular a) where
  type Diff (Tabular a) = Vector R

  d₁ .-. d₂ = toVector d₁ - toVector d₂

  d .+ v = d { mapping = HashMap.unionWith (+) (mapping d) (mapping dᵥ) }
    where dᵥ = fromVector (domain d) v

instance (Eq a, Hashable a) => Approx Tabular a where
  eval Tabular { mapping } x = mapping ! x

  update d@Tabular { mapping } xs ys = d
    { mapping = HashMap.fromList pairs <> mapping
    }
    where pairs = [ (x, y) | x <- V.toList xs | y <- Matrix.toList ys ]

  within ϵ d₁ d₂ =
    (domain d₁ == domain d₂) && Matrix.allWithin ϵ (toVector d₁) (toVector d₂)

-- | Create a 'Tabular' function approximation for the given set of
-- inputs, all initialized to 0.
create :: (Eq a, Hashable a) => [a] -> Tabular a
create xs = Tabular { mapping = HashMap.fromList [ (x, 0) | x <- xs ]
                    , domain  = V.fromList xs
                    }