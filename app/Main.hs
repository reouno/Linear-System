module Main where

import Lib
import Data.List (foldl')

main :: IO ()
main = do
    putStrLn "This is sample of learning by linear system."
    putStrLn "Learn parameter \"a\" of the function \"f(x) = ax\".\n"
    let xys = [(1,1.0),(2,2.5),(3,2.5),(4,4.5),(5,4.5)]
    let a_init = 2
    putStrLn $ "Training data = " ++ (show xys)
    putStrLn $ "Initial value of \"a\" = " ++ (show a_init)
    putStrLn "\n---------------------Batch learning---------------------\n"
    putStrLn "-----Use defined differential-----"
    a_defined <- batch_learn_def_IO f f' a_init xys 0
    putStrLn $ "Learned \"a\" value: a = " ++ (show a_defined)
    putStrLn "-----Use computed approximation of differential-----"
    a_approximate <- batch_learn_IO f a_init xys 0
    putStrLn $ "Learned \"a\" value: a = " ++ (show a_approximate)
    putStrLn "\n------------------Sequential learning------------------\n"
    a_seq <- sequential_learn_IO f_seq a_init xys 0
    putStrLn $ "Learned \"a\" value: a = " ++ (show a_seq)


-- 微分係数を計算するときにxに与える微小変化
h :: Double
h = 1.0e-7

-- 勾配法で使う、正の定数
epsilon :: Double
epsilon = 0.01


{-
--------------------
-- Batch learning --
--------------------
-}

-- Residual sum of squares
-- Cost function
f :: Num a => a -> [(a, a)] -> a
f a xys = sum $ zipWith (\x y -> (y-a*x)^2) xs ys
    where (xs, ys) = unzip xys

-- 最初から導関数を定義しておいた場合
f' :: Num a => a -> [(a, a)] -> a
f' a xys = (-2) * (sum $ zipWith (\x y -> (y-a*x)*x) xs ys)
    where (xs, ys) = unzip xys

-- 極小点探索（あらかじめ定義しておいた導関数を使う場合）
batch_learn_def :: Ord a => (Double -> t -> a) -> (Double -> t -> Double) -> Double -> t -> Double
batch_learn_def f f' a xys =
    let
        slope = f' a xys
        a' = a - epsilon * slope
    in
        if f a xys <= f a' xys
            then a
            else batch_learn_def f f' a' xys

-- 毎回近似誤差を出力
batch_learn_def_IO :: (Num a, Ord a1, Show a, Show a1)
               => (Double -> t -> a1)
               -> (Double -> t -> Double)
               -> Double
               -> t
               -> a
               -> IO Double
batch_learn_def_IO f f' a xys n = do
    let
        slope = f' a xys
        a' = a - epsilon * slope
    putStrLn $ "Approximation error: " ++ (show $ f a xys) ++ "（Training times: " ++ (show n) ++ "）"
    if f a xys <= f a' xys
        then return a
        else batch_learn_def_IO f f' a' xys (n+1)

-- 導関数を差分により近似する場合（プログラムで計算する場合）
differential :: (Double -> t -> Double) -> Double -> t -> Double
differential f a xys = (f (a+h) xys - f (a-h) xys) / (2*h)

-- 極小点探索（導関数を差分で近似計算する場合）
batch_learn :: (Double -> t -> Double) -> Double -> t -> Double
batch_learn f a xys =
    let
        slope = differential f a xys
        a' = a - epsilon * slope
    in
        if f a xys <= f a' xys
            then a
            else batch_learn f a' xys

-- 毎回近似誤差を出力
batch_learn_IO :: (Num a, Show a)
                => (Double -> t -> Double)
                -> Double
                -> t
                -> a
                -> IO Double
batch_learn_IO f a xys n = do
    let
        slope = differential f a xys
        a' = a - epsilon * slope
    putStrLn $ "Approximation error: " ++ (show $ f a xys) ++ "（Training times: " ++ (show n) ++ "）"
    if f a xys <= f a' xys
        then return a
        else batch_learn_IO f a' xys (n+1)


{-
-------------------------
-- Sequential learning --
-------------------------
-}

-- Cost function
f_seq :: Num b => b -> (b, b) -> b
f_seq a xy = (snd xy - a * fst xy)^2

-- 微分を差分による近似として計算する
diff_seq :: (Double -> t -> Double) -> Double -> t -> Double
diff_seq f a xy = (f (a+h) xy - f (a-h) xy) / (2*h)

--diff_seq' :: Num a => a -> (a,a) -> a
--diff_seq' a xy = (-2) * (snd xy - a * fst xy) * fst xy

-- 学習データを1周する
learn_data :: Eq t => (Double -> t -> Double) -> Double -> [t] -> Double
learn_data f a (xy:xys) =
    let
        slope = diff_seq f a xy
        a' = a - epsilon * slope
    in
        if xys == []
            then a'
            else learn_data f a xys

sequential_learn :: Eq t => (Double -> t -> Double) -> Double -> [t] -> Double
sequential_learn f a xys =
    let
        a' = learn_data f a xys
    in
        if (sum $ map (f a) xys) <= (sum $ map (f a') xys)
            then a
            else sequential_learn f a' xys

sequential_learn_IO f a xys n = do
    let a' = learn_data f a xys
    putStrLn $ "Approximation error: " ++ (show $ sum $ map (f a) xys) ++ " (Training times: " ++ (show n) ++ ")"
    if (sum $ map (f a) xys) <= (sum $ map (f a') xys)
        then return a
        else sequential_learn_IO f a' xys (n+1)