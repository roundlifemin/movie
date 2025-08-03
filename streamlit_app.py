import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


    



# 데이터 로딩
@st.cache_data
def data_load():
    movies = pd.read_csv('./source/m1/movies.dat', delimiter='::', header=None, engine='python', encoding='ISO-8859-1',
                         names=['MovieID', 'Title', 'Genres'])
    users = pd.read_csv('./source/m1/users.dat', sep='::', engine='python', header=None,
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    ratings = pd.read_csv('./source/m1/ratings.dat', sep='::', engine='python', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    return movies, users, ratings

# merge
@st.cache_data
def data_merge(movies, users, ratings):
    data = ratings.merge(users).merge(movies)
    recommendation_data = data[['UserID', 'MovieID', 'Rating']]
    return data, recommendation_data

# pivot
@st.cache_data
def data_pivot_corr(recommendation_data):
    pivot = recommendation_data.pivot(index='UserID', columns='MovieID', values='Rating')
    pivot.fillna(0, inplace=True)
    return pivot

# 유사 사용자
def nearest_user(corr_matrix, user_id, n):
    return corr_matrix.loc[user_id].sort_values(ascending=False)[1:n+1]

# 본 영화 목록
def movie_seen(recommendation_pivot, user_id, movies):
    seen = recommendation_pivot.loc[user_id][recommendation_pivot.loc[user_id] > 0]
    return movies[movies['MovieID'].isin(seen.index)].assign(MyRating=seen.values)

# 추천
def recommend_movie(pivot, data, movies, user_id, n=2):
    corr = pivot.T.iloc[:500, :500].corr()
    similar_users = nearest_user(corr, user_id, n).index
    sim_user_corr = nearest_user(corr, user_id, n)
    similar_data = data[(data.UserID.isin(similar_users)) & (data.Rating == 5)]
    seen = pivot.loc[user_id][pivot.loc[user_id] > 0]
    unseen = set(similar_data['MovieID']) - set(seen.index)
    return movies[movies['MovieID'].isin(unseen)].reset_index(drop=True), sim_user_corr



# main
def main():

   font_dirs = [os.getcwd() + '/Nanum_Gothic']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    
    fm._load_fontmanager(try_read_cache=False)
    fontNames = [f.name for f in fm.fontManager.ttflist]
    fontname = st.selectbox("폰트 선택", unique_font(fontNames))
    
    plt.rc('font', family=fontname)


    st.title("사용자 기반 영화 추천 시스템")
    st.markdown("**유사 사용자 기반 협업 필터링**으로 추천합니다.")
    
     # 폰트 설정 (한글 깨짐 방지)
    set_korean_font()
    
    with st.spinner("데이터 로딩 중..."):
        movies, users, ratings = data_load()
        full_data, recommendation_data = data_merge(movies, users, ratings)
        pivot = data_pivot_corr(recommendation_data)

    user_id = st.selectbox("사용자 선택", pivot.index.tolist())
    top_n = st.slider("추천받을 영화 수", 1, 10, 3)

    if st.button("영화 추천받기"):
        with st.spinner("추천 처리 중..."):
            recommended_movies, sim_user_corr = recommend_movie(pivot, recommendation_data, movies, user_id, top_n)
            seen_movies = movie_seen(pivot, user_id, movies)

        st.subheader("내가 본 영화 목록")
        st.dataframe(seen_movies[['Title', 'Genres', 'MyRating']].sort_values('MyRating', ascending=False))

        st.subheader("유사 사용자 Top-N")
        st.table(pd.DataFrame({
            "UserID": sim_user_corr.index,
            "상관계수": sim_user_corr.values
        }))

        st.subheader("유사 사용자 상관계수 시각화")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=sim_user_corr.values, y=sim_user_corr.index, ax=ax)
        ax.set_xlabel("상관계수")
        ax.set_ylabel("UserID")
        ax.set_title("Top-N 유사 사용자 상관계수")
        st.pyplot(fig)

        st.subheader("추천 영화 목록")
        st.dataframe(recommended_movies[['Title', 'Genres']])

if __name__ == '__main__':
    main()
