{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.mllib.recommendation import ALS\n",
    "import math\n",
    "ratings_rdd = sc.textFile(\"C:\\Users\\himanshu verma\\Documents\\spark\\ml-latest-small\\ml-latest-small\\ratings.csv\").map(lambda line:line.split(\",\")).map(lambda word:(word[0],word[1],word[2]))\n",
    "movies_rdd = sc.textFile(\"C:\\Users\\himanshu verma\\Documents\\spark\\ml-latest-small\\ml-latest-small\\movies.csv\").map(lambda line:line.split(\",\")).map(lambda word:(word[0],word[1]))\n",
    "training_RDD, validation_RDD, test_RDD = small_ratings_rdd.randomSplit([6, 2, 2], seed=0L)\n",
    "\n",
    "validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))\n",
    "test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))\n",
    "\n",
    "seed = 5L\n",
    "iterations = 10\n",
    "regularization_parameter = 0.1\n",
    "ranks = [4, 8, 12]\n",
    "errors = [0, 0, 0]\n",
    "err = 0\n",
    "tolerance = 0.02\n",
    "\n",
    "min_error = float('inf')\n",
    "best_rank = -1\n",
    "best_iteration = -1\n",
    "for rank in ranks:\n",
    "    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,\n",
    "                      lambda_=regularization_parameter)\n",
    "    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))\n",
    "    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)\n",
    "    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())\n",
    "    errors[err] = error\n",
    "    err = err+1\n",
    "    print 'For rank %s the RMSE is %s' % (rank, error)\n",
    "    if error < min_error:\n",
    "        min_error = error\n",
    "        best_rank = rank\n",
    "\n",
    "print 'The best model was trained with rank %s' % best_rank\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "For rank 4 the RMSE is 0.973681878574\n",
    "For rank 8 the RMSE is 0.9706475933\n",
    "For rank 12 the RMSE is 0.9601647563632\n",
    "The best model was trained with rank 8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,\n",
    "                      lambda_=regularization_parameter)\n",
    "predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))\n",
    "rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)\n",
    "error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())\n",
    "    \n",
    "print ' the RMSE is %s' , (error)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    " the RMSE is 0.982342381898"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "new_ratings = [(0,260,4), # Star Wars (1977)\n",
    "               (0,1,3), # Toy Story (1995)\n",
    "     (0,16,3), # Casino (1995)\n",
    "     (0,25,4), # Leaving Las Vegas (1995)\n",
    "     (0,32,4), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)\n",
    "     (0,335,1), # Flintstones, The (1994)\n",
    "     (0,379,1), # Timecop (1994)\n",
    "     (0,296,3), # Pulp Fiction (1994)\n",
    "     (0,858,5) , # Godfather, The (1972)\n",
    "     (0,50,4) # Usual Suspects, The (1995)\n",
    "    ]\n",
    "new_user_ratings_RDD = sc.parallelize(new_user_ratings)\n",
    "\n",
    "complete_data_with_new_ratings_RDD = ratings_rdd.union(new_user_ratings_RDD)\n",
    "new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)\n",
    "new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
