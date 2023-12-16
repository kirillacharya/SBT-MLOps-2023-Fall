#include "LinearAlgebra.h"
#include <gtest/gtest.h>

TEST(LinearAlgebraTests, DotProductBlas) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};
  EXPECT_DOUBLE_EQ(32.0, LinearAlgebra::dotBlas(a, b));
}

TEST(LinearAlgebraTests, MatrixMultiplicationBlas) {
  std::vector<std::vector<double>> matrix_a = {{1.0, 2.0}, {3.0, 4.0}};
  std::vector<std::vector<double>> matrix_b = {{2.0, 0.0}, {1.0, 3.0}};
  auto result = LinearAlgebra::matmulBlas(matrix_a, matrix_b);
  EXPECT_DOUBLE_EQ(result[0][0], 4.0);
  EXPECT_DOUBLE_EQ(result[0][1], 6.0);
  EXPECT_DOUBLE_EQ(result[1][0], 10.0);
  EXPECT_DOUBLE_EQ(result[1][1], 12.0);
}

TEST(LinearAlgebraTests, cosineSimilarityBlas) {
  std::vector<std::vector<double>> vec_set_a = {{3, 5, 7, 9},{9, 11, 13, 15}};
  std::vector<std::vector<double>> vec_set_b = {{15, 17, 19, 21},{21, 23, 25, 27}};

  auto result = LinearAlgebra::cosine_similarity(vec_set_a, vec_set_b);

  EXPECT_DOUBLE_EQ(result[0], 0.9729);  //torch.nn.functional.cosine_similarity(vec_set_a[0], vec_set_b[0], eps=1e-8)
  EXPECT_DOUBLE_EQ(result[1], 0.9958);  //torch.nn.functional.cosine_similarity(vec_set_a[1], vec_set_b[1], eps=1e-8)
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
