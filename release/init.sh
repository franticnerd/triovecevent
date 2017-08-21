cd ..

case "$(uname -s)" in
  Darwin)
    echo 'Link data path for Mac OS'
    ln -s ~/data/projects/triovecevent ./data
    ;;
  Linux)
    echo 'Link data path for Linux'
    ln -s /shared/data/czhang82/projects/triovecevent ./data
    ;;
esac

# cp /Users/chao/Dropbox/Research/base/Project/.gitignore .

# git init
# git add .gitignore
# git commit -m "first commit"
# git remote add origin https://github.com/franticnerd/triovecevent.git
# git push -u origin master
